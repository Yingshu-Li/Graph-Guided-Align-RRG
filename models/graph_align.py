import os
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from evalcap.bleu.bleu import Bleu
from evalcap.rouge.rouge import Rouge
from evalcap.cider.cider import Cider
from evalcap.meteor.meteor import Meteor
from transformers import SwinModel
from lightning_tools.optim import config_optimizer
from peft import get_peft_model, LoraConfig, TaskType
from utils import LayerNormFp32
import torch.distributed as dist
from torch.nn import functional as F
import numpy as np
import copy
from graph import TagEncoder, MultiHeadedAttention, PositionwiseFeedForward, DecoderLayer, Decoder, Encoder, EncoderLayer

class AttentionPooling(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
        nn.Linear(in_dim, in_dim),
        nn.LayerNorm(in_dim),
        nn.GELU(),
        nn.Linear(in_dim, 1),
        )

    def forward(self, last_hidden_state, attention_mask=None):
        w = self.attention(last_hidden_state).float()
        if attention_mask is not None:
            w[attention_mask==0]=float('-inf')
        w = torch.softmax(w,1)
        attention_embeddings = torch.sum(w * last_hidden_state, dim=1)
        return attention_embeddings
    

class network(pl.LightningModule):
    """
    R2GenGPT model.
    """
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(f'Loading vision encoder:{args.vision_model}')
        self.visual_encoder = SwinModel.from_pretrained(args.vision_model)
        
        print(f'Loading Trainable vision encoder:{args.vision_model} -- Done')

        print('Loading LLAMA')
        self.llama_tokenizer = AutoTokenizer.from_pretrained(args.llama_model, use_fast=False, clean_up_tokenization_spaces=False)
        self.llama_tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
        self.llama_model = AutoModelForCausalLM.from_pretrained(
            args.llama_model,
            torch_dtype="auto",
        )
        self.llama_model.config.pad_token_id = self.llama_tokenizer.pad_token_id # updating model config
         
        if args.llm_use_lora:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, 
                target_modules=["q_proj", "v_proj", "k_proj", "gate_proj", "down_proj", "up_proj", "o_proj"],
                inference_mode=False, 
                r=args.llm_r, 
                lora_alpha=args.llm_alpha, 
                lora_dropout=args.lora_dropout
            )
            self.llama_model = get_peft_model(self.llama_model, peft_config)
            self.llama_model.print_trainable_parameters()
            print('Loading LLAMA LoRA Done')         
        else:
            self.embed_tokens = self.llama_model.get_input_embeddings()
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
            print('Loading LLAMA Done')

        self.llama_proj = nn.Sequential(
            nn.Linear(self.visual_encoder.num_features, self.llama_model.config.hidden_size),
            nn.GELU(),
            nn.Linear(self.llama_model.config.hidden_size, self.llama_model.config.hidden_size),
            nn.LayerNorm(self.llama_model.config.hidden_size)
        )
        self.end_sym = args.end_sym
        # self.end_sym = self.llama_tokenizer.eos_token  # '<|eot_id|>'

        self.img_attn_pooling = AttentionPooling(self.llama_model.config.hidden_size)
        self.embed_attn_pooling = AttentionPooling(self.llama_model.config.hidden_size)
        self.logit_attn_pooling = AttentionPooling(self.llama_model.config.hidden_size)
        self.graph_img_attn_pooling = AttentionPooling(1024)
        self.graph_txt_attn_pooling = AttentionPooling(1024)

        self.img_contrastive_proj = nn.Linear(self.llama_model.config.hidden_size, 1024)
        self.txt_contrastive_proj = nn.Linear(self.llama_model.config.hidden_size, 1024)
        self.logit_contrastive_proj = nn.Linear(self.llama_model.config.hidden_size, 1024)

        self.graph_img_proj = nn.Linear(1024, 768)
        self.graph_txt_proj = nn.Linear(1024, 768)
        self.logit_scale1 = torch.nn.Parameter(torch.ones(1) * np.log(1 / 0.07), requires_grad=True)
        self.logit_scale2 = torch.nn.Parameter(torch.ones(1) * np.log(1 / 0.07), requires_grad=True)
        self.logit_scale3 = torch.nn.Parameter(torch.ones(1) * np.log(1 / 0.07), requires_grad=True)
        
        # graph
        # downsampling
        self.node_init_down = nn.Linear(self.llama_model.config.hidden_size, 1024)
        self.txt_embed_down = nn.Linear(self.llama_model.config.hidden_size, 1024)
        # self.img_embed_down = nn.Linear(self.llama_model.config.hidden_size, 1024)

        self.graph_encoder = TagEncoder(1024, 0.1)
        # self.node_init = torch.load('/mnt/sdb/yingshu/itm/node_new_25.pt')
        self.node_init = torch.load('/mnt/sdb/yingshu/itm/node_txt.pt')
        c = copy.deepcopy
        self.cross_attn_img = Decoder(DecoderLayer(1024, c(MultiHeadedAttention(8, 1024)), c(PositionwiseFeedForward(1024, 2048, 0.1)), 0.1), 3)
        self.cross_attn_txt = Decoder(DecoderLayer(1024, c(MultiHeadedAttention(8, 1024)), c(PositionwiseFeedForward(1024, 2048, 0.1)), 0.1), 3)
        self.kl = nn.KLDivLoss(reduction='batchmean', log_target=True)
        self.prompt = 'Generate a comprehensive and detailed diagnosis report for this chest xray image.'
        self.val_step_outputs = []
        self.test_step_outputs = []
        self.val_score = 0.0

        if args.delta_file is not None:
            state_dict = torch.load(args.delta_file, map_location=torch.device(f'cuda:{torch.cuda.current_device()}'))['model']
            self.load_state_dict(state_dict=state_dict, strict=False)
            print(f'Load checkpoint from {args.delta_file}')
        
        if args.freeze_vm:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            print(f'Frozen vision encoder:{args.vision_model} -- Done')


    def score(self, ref, hypo):
        """
        ref, dictionary of reference sentences (id, sentence)
        hypo, dictionary of hypothesis sentences (id, sentence)
        score, dictionary of scores
        """
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Rouge(), "ROUGE_L"),
            (Meteor(), "METEOR"),
            (Cider(), "CIDEr")
        ]
        final_scores = {}
        for scorer, method in scorers:
            score, scores = scorer.compute_score(ref, hypo)
            if type(score) == list:
                for m, s in zip(method, score):
                    final_scores[m] = s
            else:
                final_scores[method] = score
        return final_scores


    def encode_img(self, images):
        image_embeds = []
        for image in images:
            device = image.device
            if self.hparams.global_only:
                image_embed = self.visual_encoder(image)['pooler_output'].unsqueeze(1).to(device)
            else:
                image_embed = self.visual_encoder(image)['last_hidden_state'].to(device)
            image_embeds.append(image_embed)
            
        image_embeds = torch.stack(image_embeds).mean(0)
        atts_llama = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)
        return image_embeds, atts_llama
    

    def encode_txt(self, text):
        device = self.llama_model.device
        tokens = self.llama_tokenizer(text, return_tensors="pt", padding="longest", truncation=True, max_length=self.hparams.max_length, add_special_tokens=False).to(device)
        atten_mask = tokens.attention_mask
        text_embeds = self.embed_tokens(tokens.input_ids)  # bs * seq_len * 4096
        # get the mean of the text embeds except the padding tokens
        mean_text_embeds = self.embed_attn_pooling(text_embeds, atten_mask)
        return mean_text_embeds, text_embeds


    def prompt_wrap(self, img_embeds, atts_img):
        prompt=f'<|start_header_id|>user<|end_header_id|> <Img><ImageHere></Img> {self.prompt} <|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n'
        batch_size = img_embeds.shape[0]
        img_embeds = self.llama_proj(img_embeds)
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        image_begin_index = p_before_embeds.shape[1]
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        return wrapped_img_embeds, wrapped_atts_img, image_begin_index
    
       
    def forward(self, samples):
        image = samples["image"]
        report = samples["input_text"]
        img_embeds_, atts_img = self.encode_img(image)
        # graph related
        mean_txt_embeds, txt_embeds = self.encode_txt(report) 
        node_init = self.node_init.to(img_embeds_.device)
        node_init = self.node_init_down(node_init)
        node_init = node_init.expand(img_embeds_.shape[0], -1, -1)
        graph_embeds = self.graph_encoder(node_init)

        img_embed_down = img_embeds_ #self.img_embed_down(img_embeds_)
        txt_embeds = self.txt_embed_down(txt_embeds)
        vis_graph_embeds, _ = self.cross_attn_img(graph_embeds, img_embed_down)
        txt_graph_embeds, _ = self.cross_attn_txt(graph_embeds, txt_embeds)

        # cat_feats = torch.cat([img_embeds_, vis_graph_embeds], dim=1)
        # cat_feats = self.transformer_encoder(cat_feats)

        img_embeds, atts_img, image_begin_index = self.prompt_wrap(img_embeds_, atts_img)

        self.llama_tokenizer.padding_side = "right"
        text = [t + self.end_sym for t in samples["input_text"]]

        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        ).to(image[0].device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        empty_targets = (
            torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                       dtype=torch.long).to(image[0].device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=to_regress_tokens.input_ids.dtype,
                         device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        to_regress_embeds = self.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

        outputs = self.llama_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            labels=targets,
        )

        # image_embed += bos_token
        image_begin_index = image_begin_index + 1
        # mean_txt_embeds, txt_embeds = self.encode_txt(report) 
        img_after_llama = outputs.hidden_states[-1][:, image_begin_index: image_begin_index + 50, :]
        img_after_llama_pooling = self.img_attn_pooling(img_after_llama)

        # global contrastive loss between image after llama and word embeddings
        img_after_llama_pooling = self.img_contrastive_proj(img_after_llama_pooling)
        mean_txt_embeds = self.txt_contrastive_proj(mean_txt_embeds)
        img_after_llama_pooling = F.normalize(img_after_llama_pooling, p=2, dim=1)
        mean_txt_embeds = F.normalize(mean_txt_embeds, p=2, dim=1)
        logits_per_image, logits_per_text = self.get_logits(img_after_llama_pooling, mean_txt_embeds, self.logit_scale1)
        contrastive_loss_1 = self._clip_loss(logits_per_image, logits_per_text)

        labels = samples["label"]
        # labels[labels==2] = -1
        labels_expanded = labels.unsqueeze(1)
        labels_t_expanded = labels.unsqueeze(0)
        sim = (labels_expanded == labels_t_expanded).float()
        match_rates = sim.sum(2) + torch.eye(sim.shape[0], device=img_embeds.device, dtype=torch.long)
        similarity_matrix = match_rates

        # norm_same = (labels_expanded[:, :, 8] == labels_t_expanded[:, :, 8]).float()
        # penalty_value = 0.5 * match_rates
        # similarity_matrix = match_rates * norm_same + (1 - norm_same) * penalty_value
        # similarity_matrix = similarity_matrix.to(img_embeds.device)

        contrastive_loss_2 = self._soft_clip_loss(logits_per_image, similarity_matrix)
        contrastive_loss = (contrastive_loss_1 + contrastive_loss_2) / 2

        vis_graph_embeds = self.graph_img_proj(vis_graph_embeds)
        txt_graph_embeds = self.graph_txt_proj(txt_graph_embeds)

        # local loss
        node_loss_hard = 0
        node_loss_soft = 0
        for i in range(vis_graph_embeds.shape[1]):
            per_node_img = vis_graph_embeds[:, i, :]
            per_node_txt = txt_graph_embeds[:, i, :]
            per_node_img = F.normalize(per_node_img, p=2, dim=1)
            per_node_txt = F.normalize(per_node_txt, p=2, dim=1)
            logits_per_node_image, logits_per_node_text = self.get_logits(per_node_img, per_node_txt, self.logit_scale3)
            local_loss = self._clip_loss(logits_per_node_image, logits_per_node_text)
            node_loss_hard += local_loss
            label = labels[:, i].unsqueeze(1)
            label_expanded = label.unsqueeze(1)
            label_t_expanded = label.unsqueeze(0)
            sim = (label_expanded == label_t_expanded).float()
            match_rate = sim.sum(2)
            similarity_matrix = match_rate + 0.5*torch.eye(logits_per_node_image.shape[0], device=img_embeds.device, dtype=torch.long)
            # local_loss_soft = self._soft_clip_loss_local(logits_per_node_image, similarity_matrix)
            # single-modality contrastive loss
            img_sim_logits = per_node_img @ per_node_img.T
            txt_sim_logits = per_node_txt @ per_node_txt.T
            local_loss_soft = (
                self._soft_clip_loss_local(logits_per_node_image, similarity_matrix) + 
                self._soft_clip_loss_local(img_sim_logits, similarity_matrix) +
                self._soft_clip_loss_local(txt_sim_logits, similarity_matrix)
            ) / 3
            # local_loss_soft = (
            #     self._soft_clip_loss_local(logits_per_node_image, similarity_matrix) #+ 
            #     # self._soft_clip_loss_local(img_sim_logits, similarity_matrix) +
            #     # self._soft_clip_loss_local(txt_sim_logits, similarity_matrix)
            # ) #/ 3
            node_loss_soft += local_loss_soft


        # node_loss = ((node_loss_hard + node_loss_soft) / 2) / vis_graph_embeds.shape[1]
        # loss = outputs.loss
        # return {"loss": loss + node_loss, 
        #         "soft_node_loss": node_loss_soft,
        #         "hard_node_loss": node_loss_hard,
        #         "node_loss": node_loss 
        #         }

        ###### ALL APPLIED LOSS ######
        node_loss = ((node_loss_hard + node_loss_soft) / 2) / vis_graph_embeds.shape[1]

        lamb_1 = self.args.lamb_1
        lamb_2 = self.args.lamb_2
        loss = outputs.loss
        return {"loss": loss + lamb_1 * node_loss + lamb_2 * contrastive_loss, 
                "lm_loss": outputs.loss, 
                "contrastive_loss": contrastive_loss,               
                "contrastive_loss_1": contrastive_loss_1, 
                "contrastive_loss_2": contrastive_loss_2,
                "soft_node_loss": node_loss_soft,
                "hard_node_loss": node_loss_hard,
                "node_loss": node_loss 
                }

    def training_step(self, batch, batch_idx):
        result = self(batch)
        self.log_dict(result, prog_bar=True)
        return result

    def save_checkpoint(self, eval_res):
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        param_grad_dic = {
            k: v.requires_grad for (k, v) in self.named_parameters() if v.requires_grad
        }
        state_dict = self.state_dict()
        for k in list(state_dict.keys()):
            if k not in param_grad_dic.keys():
                del state_dict[k]
        save_obj = {
            "model": state_dict,
            "config": self.hparams,
            "epoch": current_epoch,
            "step":global_step
        }
        os.makedirs(os.path.join(self.hparams.savedmodel_path, 'checkpoints'), exist_ok=True)
        save_to = os.path.join(
            self.hparams.savedmodel_path, 'checkpoints',
            "checkpoint_epoch{}_step{}_bleu{:3f}_cider{:3f}.pth".format(current_epoch, global_step, eval_res['Bleu_4'], eval_res['CIDEr']),
        )
        self.print("Saving checkpoint at step {} to {}.".format(global_step, save_to))
        torch.save(save_obj, save_to)
    
    def validation_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds_, atts_img = self.encode_img(image)

         # graph related
        node_init = self.node_init.to(img_embeds_.device)
        node_init = self.node_init_down(node_init)
        node_init = node_init.expand(img_embeds_.shape[0], -1, -1)
        graph_embeds = self.graph_encoder(node_init)

        img_embed_down = img_embeds_
        # txt_embeds = self.txt_embed_down(txt_embeds)
        vis_graph_embeds, _ = self.cross_attn_img(graph_embeds, img_embed_down)
        # cat_feats = torch.cat([img_embeds_, vis_graph_embeds], dim=1)
        # cat_feats = self.transformer_encoder(cat_feats)
        img_embeds, atts_img, _ = self.prompt_wrap(img_embeds_, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.llama_tokenizer.pad_token_id
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.val_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref
    
    def decode(self, output_token):
        if output_token[0] == 0:  # the model might output a unknow token <unk> at the beginning. remove it
            output_token = output_token[1:]
        if output_token[0] == 1:  # some users find that there is a start token <s> at the beginning. remove it
            output_token = output_token[1:]
        output_text = self.llama_tokenizer.decode(output_token, add_special_tokens=False)
        output_text = output_text.split(self.end_sym)[0].strip()
        output_text = output_text.replace('<unk>', '')
        output_text = output_text.replace(self.llama_tokenizer.pad_token, '')
        output_text = output_text.replace('<|end_of_text|>', '')
        output_text = output_text.replace('<|eot_id|>', '')
        return output_text

    def on_validation_epoch_end(self):
        self.val_step_outputs = self.all_gather(self.val_step_outputs)
        ref, hypo, ids = [], [], []
        for cnt in range(len(self.val_step_outputs)):
            for i in self.val_step_outputs[cnt]:
                ref.extend(i['ref'])
                hypo.extend(i['hypo'])
                ids.extend(i['id'])

        # ref, hypo, ids = [], [], []
        # for i in self.val_step_outputs:
        #     ref.extend(i['ref'])
        #     hypo.extend(i['hypo'])
        #     ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)
        self.log_dict(eval_res, sync_dist=True, logger=True)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        current_epoch, global_step = self.trainer.current_epoch, self.trainer.global_step
        json.dump(hypo, open(os.path.join(result_folder, f"result_{current_epoch}_{global_step}" + '.json'), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'refs.json'), 'w'))
        self.print(eval_res)

        val_score = 0
        for score_type, weight in zip(self.hparams.scorer_types, self.hparams.weights):
            val_score += eval_res[score_type] * weight

        if self.trainer.local_rank == 0:
            if val_score > self.val_score:
                self.save_checkpoint(eval_res)
                self.val_score = val_score
        self.val_step_outputs.clear()


    def test_step(self, samples, batch_idx):
        self.llama_tokenizer.padding_side = "right"
        to_regress_tokens = self.llama_tokenizer(
            samples['input_text'],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hparams.max_length,
            add_special_tokens=False
        )

        image = samples["image"]
        img_embeds, atts_img = self.encode_img(image)
        img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img)

        batch_size = img_embeds.shape[0]
        bos = torch.ones([batch_size, 1],
                         dtype=atts_img.dtype,
                         device=atts_img.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        atts_bos = atts_img[:, :1]

        inputs_embeds = torch.cat([bos_embeds, img_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, atts_img], dim=1)

        outputs = self.llama_model.generate(
            inputs_embeds=inputs_embeds,
            num_beams=self.hparams.beam_size,
            do_sample=self.hparams.do_sample,
            min_new_tokens=self.hparams.min_new_tokens,
            max_new_tokens=self.hparams.max_new_tokens,
            repetition_penalty=self.hparams.repetition_penalty,
            length_penalty=self.hparams.length_penalty,
            temperature=self.hparams.temperature,
            pad_token_id=self.llama_tokenizer.eos_token_id
        )
        hypo = [self.decode(i) for i in outputs]
        ref = [self.decode(i) for i in to_regress_tokens['input_ids']]
        self.test_step_outputs.append({"hypo": hypo, "ref": ref, "id": samples["id"]})
        return hypo, ref


    def on_test_epoch_end(self):
        """
        This function is called at the end of the test epoch.
        It is recommended to test on single device to ensure each sample/batch gets evaluated exactly once. This is helpful to make sure benchmarking for research papers is done the right way. Otherwise, in a multi-device setting, samples could occur duplicated when DistributedSampler is used, for eg. with strategy="ddp". It replicates some samples on some devices to make sure all devices have same batch size in case of uneven inputs.
        """
        ref, hypo, ids = [], [], []
        for i in self.test_step_outputs:
            ref.extend(i['ref'])
            hypo.extend(i['hypo'])
            ids.extend(i['id'])

        ref = {k:[v] for k, v in zip(ids, ref)}
        hypo = {k:[v] for k, v in zip(ids, hypo)}
        eval_res = self.score(ref=ref,hypo=hypo)

        result_folder = os.path.join(self.hparams.savedmodel_path, 'result')
        os.makedirs(result_folder, exist_ok=True)
        json.dump(hypo, open(os.path.join(result_folder, f"test_result.json"), 'w'))
        json.dump(ref, open(os.path.join(result_folder, 'test_refs.json'), 'w'))
        self.print(f"Test result of {self.hparams.delta_file}: {eval_res}")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.hparams.max_epochs-1, eta_min=1e-6)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):

        optimizer.step(closure=optimizer_closure)
        num_warmup_step = 500.0
        if self.trainer.global_step < num_warmup_step:
            lr_scale = min(1.0, float(self.trainer.global_step + 1) / num_warmup_step)
            for pg in optimizer.param_groups:
                pg["lr"] = lr_scale * self.hparams.learning_rate


    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items


    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad()
    

    @torch.no_grad()
    def all_gather(self, data):
        """
        Performs all_gather operation on the provided tensors.
        *** Warning ***: torch.distributed.all_gather has no gradient.
        """
        dist.barrier()
        gather_data = [None for _ in range(torch.distributed.get_world_size())]
        dist.all_gather_object(gather_data, data)
        return gather_data
    

    def get_logits(self, image_features, text_features, logit_scale):
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text
    

    def _clip_loss(self, logits_per_image, logits_per_text):
        device = logits_per_image.device
        labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long) #self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2
        return total_loss
    

    def _soft_clip_loss(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        image_loss = self._soft_xent_loss(F.softmax(logits_per_img, 1), F.softmax(soft_label, 1))
        # caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        # return (image_loss + caption_loss) / 2


        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        return image_loss
    
    def _soft_clip_loss_local(self, logits_per_img, soft_label):
        '''take labels of images and sentences as a softlabel
        e.g., image_label = [1, 0, 1, -1], sentence_label = [0, 0, 1, -1]
        this pair has similarity as: 1 * 0 + 0 * 0 + 1 * 1 + -1 * -1 = 2.
        We will clamp the similarity into [-1,1], and take softmax as a soft-label.
        '''
        # when using InfoNCE-like loss
        soft_label = (soft_label / soft_label.sum(1, keepdim=True)) + 1e-8
        image_loss = self._soft_xent_loss(F.softmax(logits_per_img, 1), soft_label)
        # caption_loss = self._soft_xent_loss(logits_per_img.T, F.softmax(soft_label.T,1))
        # return (image_loss + caption_loss) / 2


        # when using multilabel bce loss
        # image_loss = self._soft_bce_loss(logits_per_img, soft_label)
        return image_loss
    

    def _soft_xent_loss(self, p, q):
        # https://discuss.pytorch.org/t/jensen-shannon-divergence/2626/10
        p, q = p.view(-1, p.size(-1)), q.view(-1, q.size(-1))
        m = (0.5 * (p + q)).log()
        return 0.5 * (self.kl(m, p.log()) + self.kl(m, q.log()))

    def _soft_bce_loss(self, input, target):
        return nn.functional.binary_cross_entropy_with_logits(input, target)