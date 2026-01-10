"""
HERO Model
面向多模态情感理解的分层式证据推理与观察框架
Refactored for Phase 1 Alignment (Text Expert, Audio-Guided Attention)

This module implements the full HERO model combining:
1. Observation Expert Layer
2. Evidence Integration Layer  
3. Hierarchical Reasoning Layer (LLM)

Reference: HERO Idea.md
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from minigpt4.common.registry import registry
from minigpt4.models.minigpt_base import MiniGPTBase
from minigpt4.models.base_model import disabled_train
from minigpt4.models.hero.observation_experts import ObservationExpertLayer, ExpertOutput
from minigpt4.models.hero.integration_layer import EvidenceIntegrationLayer, IntegrationOutput
from minigpt4.models.hero.hero_loss import STMIL_Loss, SCCL_Loss


@dataclass
class HEROOutput:
    """Output from HERO model."""
    loss: torch.Tensor
    logits: Optional[torch.Tensor] = None
    integrated_context: Optional[torch.Tensor] = None
    attention_weights: Optional[Dict[str, torch.Tensor]] = None
    generated_text: Optional[List[str]] = None
    semantic_evidence: Optional[Dict[str, List[str]]] = None # Text evidence from experts


@registry.register_model("hero")
class HEROModel(MiniGPTBase):
    """
    HERO: Hierarchical Evidence-based Reasoning and Observation
    """
    
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/hero.yaml",
    }
    
    COT_PROMPT_TEMPLATE = (
        "Based on the multimodal evidence provided (Visual, Audio, Physiological), analyze the emotion of the subject.\n\n"
        "Please structure your response as follows:\n"
        "1. **Evidence Analysis**: Briefly describe key cues from each modality.\n"
        "2. **Rationale**: Explain how these cues interact (e.g., match, conflict) to support your conclusion.\n"
        "3. **Prediction**: State the final emotion label and your confidence.\n\n"
        "Output Format:\n"
        "### Evidence Analysis\n"
        "- Visual: ...\n"
        "- Audio: ...\n"
        "- Physiological: ...\n\n"
        "### Rationale\n"
        "...\n\n"
        "### Prediction\n"
        "{\n"
        '  "emotion": "...",\n'
        '  "confidence": ...\n'
        "}"
    )
    
    def __init__(
        self,
        vit_model: str = "eva_clip_g",
        img_size: int = 448,
        drop_path_rate: float = 0,
        use_grad_checkpoint: bool = False,
        vit_precision: str = "fp16",
        freeze_vit: bool = True,
        llama_model: str = "",
        prompt_template: str = '[INST] {} [/INST]',
        max_txt_len: int = 300,
        end_sym: str = '\n',
        lora_r: int = 64,
        lora_target_modules: List[str] = ["q_proj", "v_proj"],
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        chat_template: bool = False,
        use_grad_checkpoint_llm: bool = False,
        max_context_len: int = 3800,
        low_resource: bool = False,
        device_8bit: int = 0,
        # HERO specific
        visual_dim: int = 1408,
        video_dim: int = 1024,
        audio_dim: int = 1024,
        au_dim: int = 1024,
        text_dim: int = 768, # Added
        hidden_dim: int = 768,
        num_queries: int = 32,
        num_qformer_layers: int = 2,
        num_output_queries: int = 64,
        include_synergy: bool = True,
        use_observation_experts: bool = True,
        use_integration_layer: bool = True,
        # Loss weights
        lambda_stmil: float = 0.1,
        lambda_sccl: float = 0.1,
    ):
        super().__init__(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            max_txt_len=max_txt_len,
            max_context_len=max_context_len,
            end_sym=end_sym,
            prompt_template=prompt_template,
            low_resource=low_resource,
            device_8bit=device_8bit,
            lora_r=lora_r,
            lora_target_modules=lora_target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        
        self.hidden_dim = hidden_dim
        self.use_observation_experts = use_observation_experts
        self.use_integration_layer = use_integration_layer
        self.chat_template = chat_template
        
        # Get LLM hidden dimension
        llm_hidden_dim = self.llama_model.config.hidden_size
        
        # Original projectors from MiniGPTv2 (for backward compatibility)
        img_f_dim = self.visual_encoder.num_features * 4
        self.llama_proj = nn.Linear(img_f_dim, llm_hidden_dim)
        
        self.feats_llama_proj1 = nn.Linear(1024, llm_hidden_dim)
        self.feats_llama_proj2 = nn.Linear(1024, llm_hidden_dim)
        self.feats_llama_proj3 = nn.Linear(1024, llm_hidden_dim)
        self.feats_llama_proj4 = nn.Linear(1024, llm_hidden_dim)
        
        self.cls_tk_llama_proj = nn.Linear(1408, llm_hidden_dim)
        
        # HERO Observation Expert Layer
        if use_observation_experts:
            self.observation_experts = ObservationExpertLayer(
                visual_dim=visual_dim,
                video_dim=video_dim,
                audio_dim=audio_dim,
                au_dim=au_dim,
                text_dim=text_dim,
                hidden_dim=hidden_dim,
                num_queries=num_queries,
                num_qformer_layers=num_qformer_layers,
                llm_hidden_dim=llm_hidden_dim,
                include_synergy=include_synergy,
            )
        
        # HERO Evidence Integration Layer
        if use_integration_layer:
            self.integration_layer = EvidenceIntegrationLayer(
                hidden_dim=hidden_dim,
                llm_dim=llm_hidden_dim,
                num_heads=8,
                num_output_queries=num_output_queries,
                max_experts=6, # GlobalVis, Motion, Audio, AU, Text, Synergy
            )
        
        # HERO Auxiliary Losses
        self.stmil_loss = STMIL_Loss(feature_dim=hidden_dim)
        self.sccl_loss = SCCL_Loss()
        self.lambda_stmil = lambda_stmil
        self.lambda_sccl = lambda_sccl
        
        # Final context projection
        self.hero_context_proj = nn.Linear(llm_hidden_dim, llm_hidden_dim)
        
        if use_grad_checkpoint_llm:
            self.llama_model.gradient_checkpointing_enable()
            
    def encode_img_legacy(self, image: torch.Tensor, video_features: torch.Tensor):
        """
        Legacy encoding method (compatible with original Emotion-LLaMA).
        """
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])
            
        with self.maybe_autocast():
            image_feats = self.visual_encoder(image)
            image_embeds = self.ln_vision(image_feats).to(device)
            image_cls_tk = image_embeds[:, :1, :]
            cls_tk_feats = self.cls_tk_llama_proj(image_cls_tk)
            image_embeds = image_embeds[:, 1:, :]
            bs, pn, hs = image_embeds.shape
            image_embeds = image_embeds.view(bs, int(pn / 4), int(hs * 4))
            image_inputs_llama = self.llama_proj(image_embeds)
            
            video_features = video_features.to(device)
            num_modalities = video_features.size(1)
            video_features_split = torch.split(video_features, 1, dim=1)
            
            proj_layers = [
                self.feats_llama_proj1,
                self.feats_llama_proj2,
                self.feats_llama_proj3,
            ]
            if num_modalities == 4:
                proj_layers.append(self.feats_llama_proj4)
                
            projected_feats = []
            for idx, feats in enumerate(video_features_split):
                if idx >= len(proj_layers):
                    break
                projected_feats.append(proj_layers[idx](feats.squeeze(1)))
                
            video_feats = torch.stack(projected_feats, dim=1)
            inputs_llama = torch.cat((image_inputs_llama, video_feats, cls_tk_feats), dim=1)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
            
        return inputs_llama, atts_llama
    
    def encode_img_hero(
        self,
        image: torch.Tensor,
        video_features: torch.Tensor,
        return_expert_outputs: bool = False,
    ):
        """
        HERO encoding method using observation experts and integration layer.
        """
        device = image.device
        if len(image.shape) > 4:
            image = image.reshape(-1, *image.shape[-3:])
        video_features_dict = {}
        if video_features is not None:
            # Feature-Only Mode Logic:
            # Expecting video_features to be [B, N_Modalities, Seq, MaxDim]
            # Order Mapping from HERODataset:
            # 0: Motion (VideoMAE)
            # 1: Audio (HuBERT)
            # 2: AU
            # 3: Text
            # 4: Visual Global (CLIP)
            
            # Slice and Un-pad
            # Motion
            if video_features.shape[1] > 0:
                 f = video_features[:, 0]
                 video_features_dict['vis_motion'] = f[:, :, :self.observation_experts.video_dim]
            # Audio
            if video_features.shape[1] > 1:
                 f = video_features[:, 1]
                 video_features_dict['audio'] = f[:, :, :self.observation_experts.audio_dim]
            # AU
            if video_features.shape[1] > 2:
                 f = video_features[:, 2]
                 video_features_dict['au'] = f[:, :, :self.observation_experts.au_dim]
            # Text
            if video_features.shape[1] > 3:
                 f = video_features[:, 3]
                 video_features_dict['text'] = f[:, :, :self.observation_experts.text_dim]

            # Visual Global (Override image encoder if present)
            image_embeds = None
            if video_features.shape[1] > 4:
                 f = video_features[:, 4]
                 # CLIP features pre-extracted. Use them instead of running encoder.
                 # Shape [B, N_Patches, D].
                 # If we have this, we ignore 'image' input (which is dummy).
                 image_embeds = f[:, :, :self.observation_experts.visual_dim]
            else:
                 # Standard Mode: Run Image Encoder
                 if image is not None:
                     with self.maybe_autocast():
                         image_feats = self.visual_encoder(image)
                         image_embeds = self.ln_vision(image_feats).to(device)
                 else:
                     image_embeds = None

            video_features_dict['vis_global'] = image_embeds
            
        else:
            # Fallback for Legacy/Standard mode (if video_features is just Motion??)
            # Assuming old format [B, T, D] was just Motion?
            # Adapt if needed. For now assuming new uniform/stacked format via HERODataset.
             image_embeds = None
             if image is not None:
                 with self.maybe_autocast():
                     image_feats = self.visual_encoder(image)
                     image_embeds = self.ln_vision(image_feats).to(device)
             else:
                 image_embeds = None
             video_features_dict['vis_global'] = image_embeds
            
        # Get Expert features
        expert_outputs, all_pooled_features = self.observation_experts(
            video_features_dict['vis_global'], # Passed explicitly as 'vis_global'
            video_features_dict.get('vis_motion'),
            video_features_dict.get('audio'),
            video_features_dict.get('au'),
            video_features_dict.get('text'),
        )
        # 4. Get Summaries
        summary_vectors = self.observation_experts.get_summary_vectors(expert_outputs)
        
        # 5. Get Feature Tensors for Integration
        expert_features_map = {}
        for key, output in expert_outputs.items():
            expert_features_map[key] = output.feature_tensor
        
        # 6. Evidence Integration w/ Audio-Guided Attention
        integration_output = self.integration_layer(
            expert_features=expert_features_map,
            summary_vectors=summary_vectors,
        )
        
        # 7. Project for LLM
        hero_context = self.hero_context_proj(integration_output.integrated_context)
        
        atts_llama = torch.ones(hero_context.size()[:-1], dtype=torch.long).to(device)
        
        # Collect semantic evidence
        semantic_evidence = {}
        for k, v in expert_outputs.items():
            if v.semantic_evidence:
                semantic_evidence[k] = v.semantic_evidence

        if return_expert_outputs:
            return hero_context, atts_llama, expert_outputs, integration_output
            
        return hero_context, atts_llama
    
    def encode_img(self, image: torch.Tensor, video_features: torch.Tensor):
        """
        Main encoding method - uses HERO if available, else legacy.
        """
        if self.use_observation_experts and self.use_integration_layer:
            return self.encode_img_hero(image, video_features)
        else:
            return self.encode_img_legacy(image, video_features)
    
    def forward(self, samples: Dict[str, Any], reduction: str = 'mean') -> Dict[str, Any]:
        """
        Forward pass for training with HERO Auxiliary Losses.
        """
        # 1. Encode Images & Features (HERO Path)
        # Note: We must replicate the logic of preparing_embedding to access intermediate features
        
        # A. Encode
        if self.use_observation_experts and self.use_integration_layer:
             hero_context, atts_llama, expert_outputs, integration_output = self.encode_img_hero(
                 samples["image"], 
                 samples["video_features"],
                 return_expert_outputs=True
             )
             img_embeds, img_atts = hero_context, atts_llama
        else:
             img_embeds, img_atts = self.encode_img_legacy(samples["image"], samples["video_features"])
             expert_outputs = None
        
        # B. Prepare Prompt/Text Embeddings (Reuse Base Logic partially or calls helpers)
        # Since preparing_embedding calls encode_img internally, we can't use it directly if we want intermediates.
        # We need to manually do the text wrapping part.
        
        # Reuse internal helper if possible? 
        # Base class `preparing_embedding` calls `encode_img`. If we override `encode_img` to cache, 
        # distinct concurrent forward passes might collide.
        # Safest is to duplicate the text prep logic here or assume standard forward is enough for Task Loss
        # and re-run for Aux Loss? Re-running is expensive.
        # Let's manually do the text prep.
        
        # ... [Simplified text prep matching minigpt_base.py] ...
        # Handling conversation datasets
        if 'conv_q' in samples:
            conv_q, conv_a = samples['conv_q'], samples['conv_a']
            connect_sym = samples['connect_sym'][0]
            conv_q = [q.split(connect_sym)for q in conv_q]
            conv_a = [a.split(connect_sym) for a in conv_a]
            conv_q = [[self.prompt_template.format(item) for item in items] for items in conv_q]
            
            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, [q[0] for q in conv_q])
            regress_token_ids, regress_atts, part_targets = self.tokenize_conversation(conv_q, conv_a)
            
        else:
            # Instruction tuning case
            if "instruction_input" in samples:
                instruction = samples["instruction_input"]
            elif self.prompt_list:
                instruction = random.choice(self.prompt_list)
            else:
                instruction = None
            if hasattr(self, 'chat_template') and self.chat_template:
                instruction = [self.prompt_template.format(instruct) for instruct in instruction]
            
            cond_embeds, cond_atts = self.prompt_wrap(img_embeds, img_atts, instruction)
            
            # Target
            self.llama_tokenizer.padding_side = "right"
            text = [t + self.end_sym for t in samples["answer"]]
            regress_tokens = self.llama_tokenizer(
                text, return_tensors="pt", padding="longest", truncation=True, 
                max_length=self.max_txt_len, add_special_tokens=False
            ).to(self.device)
            regress_token_ids = regress_tokens.input_ids
            regress_atts = regress_tokens.attention_mask
            part_targets = regress_token_ids.masked_fill(
                regress_token_ids == self.llama_tokenizer.pad_token_id, -100
            )
            regress_embeds = self.embed_tokens(regress_token_ids)
        
        # C. Concat
        inputs_embeds, attention_mask, input_lens = \
            self.concat_emb_input_output(cond_embeds, cond_atts, regress_embeds, regress_atts)
            
        # D. Add BOS
        bos = torch.ones_like(part_targets[:, :1]) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.embed_tokens(bos)
        bos_atts = cond_atts[:, :1]
        inputs_embeds = torch.cat([bos_embeds, inputs_embeds], dim=1)
        attention_mask = torch.cat([bos_atts, attention_mask], dim=1)
        
        # E. Targets
        targets = torch.ones([inputs_embeds.shape[0], inputs_embeds.shape[1]],
                             dtype=torch.long).to(self.device).fill_(-100)
        for i, target in enumerate(part_targets):
            targets[i, input_lens[i]+1:input_lens[i]+len(target)+1] = target

        # F. LLaMA Forward (Task Loss)
        with self.maybe_autocast():
             outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
                reduction=reduction,
            )
        loss_task = outputs.loss
        
        # 2. Compute Auxiliary Losses (HERO)
        loss_stmil = torch.tensor(0.0).to(self.device)
        loss_sccl = torch.tensor(0.0).to(self.device)
        
        if self.use_observation_experts and expert_outputs is not None:
            # A. STMIL: Disentangle Emotion keys from Content
            # Assumption: 'audio' or 'vis_motion' contains Emotion. 'text' or 'vis_global' contains Content.
            # Let's minimize MI between Audio (Emotion) and Text (Content).
            if 'audio' in expert_outputs and 'text' in expert_outputs:
                e_feat = expert_outputs['audio'].pooled_feature
                c_feat = expert_outputs['text'].pooled_feature
                loss_stmil += self.stmil_loss(e_feat, c_feat)
                
            # B. SCCL: Align Expert Features with Semantic Evidence (Text)
            # If we have Ground Truth evidence text, we use it. 
            # Otherwise, use Text Expert features as anchor?
            # Modified: Align Visual Global <-> Text Expert (Scene Context <-> Verbal Content)
            # We avoid Audio<->Text here because STMIL is pushing them apart (disentanglement).
            if 'vis_global' in expert_outputs and 'text' in expert_outputs:
                loss_sccl += self.sccl_loss(
                    expert_outputs['vis_global'].pooled_feature,
                    expert_outputs['text'].pooled_feature
                )
                
        # Total Loss
        loss_total = loss_task + (self.lambda_stmil * loss_stmil) + (self.lambda_sccl * loss_sccl)
        
        return {
            "loss": loss_total,
            "loss_task": loss_task,
            "loss_stmil": loss_stmil,
            "loss_sccl": loss_sccl
        }
    
    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        video_features: torch.Tensor,
        texts: List[str],
        num_beams: int = 1,
        max_new_tokens: int = 512,
        min_length: int = 1,
        top_p: float = 0.9,
        repetition_penalty: float = 1.0,
        length_penalty: float = 1.0,
        temperature: float = 1.0,
        do_sample: bool = False,
        stop_words_ids: List[int] = [2],
        use_cot: bool = False,
        return_dict: bool = False,
    ) -> Any:
        # Apply CoT template if requested
        prompts = texts
        if use_cot:
            prompts = [
                self.COT_PROMPT_TEMPLATE if t.strip() == "" else t 
                for t in texts
            ]
            
        outputs = super().generate(
            images=images,
            video_features=video_features,
            texts=prompts,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
            do_sample=do_sample,
            stop_words_ids=stop_words_ids,
        )
        
        if return_dict and use_cot:
            results = []
            for output in outputs:
                parsed = self._parse_cot_output(output)
                results.append({
                    "raw_output": output,
                    "parsed_prediction": parsed
                })
            return results
            
        return outputs

    def _parse_cot_output(self, text: str) -> Dict[str, Any]:
        """
        Parse CoT output to extract structured prediction.
        """
        try:
            # Try to find JSON block in Prediction section
            if "### Prediction" in text:
                pred_section = text.split("### Prediction")[-1]
                # Look for JSON-like structure
                match = re.search(r'\{.*\}', pred_section, re.DOTALL)
                if match:
                    json_str = match.group(0)
                    return json.loads(json_str)
            
            # Fallback parsing attempt
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
                
        except Exception as e:
            pass
            
        return {"emotion": "unknown", "confidence": 0.0, "raw": text}
    
    @classmethod
    def from_config(cls, cfg):
        """
        Build model from config.
        """
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size", 448)
        llama_model = cfg.get("llama_model", "")
        
        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        low_resource = cfg.get("low_resource", False)
        
        prompt_template = cfg.get("prompt_template", '[INST] {} [/INST]')
        max_txt_len = cfg.get("max_txt_len", 300)
        end_sym = cfg.get("end_sym", '\n')
        
        lora_r = cfg.get("lora_r", 64)
        lora_alpha = cfg.get("lora_alpha", 16)
        chat_template = cfg.get("chat_template", False)
        
        use_grad_checkpoint_llm = cfg.get("use_grad_checkpoint_llm", False)
        max_context_len = cfg.get("max_context_len", 3800)
        
        # HERO specific configs
        hero_cfg = cfg.get("hero", {})
        visual_dim = hero_cfg.get("visual_dim", 1408)
        video_dim = hero_cfg.get("video_dim", 1024)
        audio_dim = hero_cfg.get("audio_dim", 1024)
        au_dim = hero_cfg.get("au_dim", 1024)
        text_dim = hero_cfg.get("text_dim", 768) # Added
        hidden_dim = hero_cfg.get("hidden_dim", 768)
        num_queries = hero_cfg.get("num_queries", 32)
        num_qformer_layers = hero_cfg.get("num_qformer_layers", 2)
        num_output_queries = hero_cfg.get("num_output_queries", 64)
        include_synergy = hero_cfg.get("include_synergy", True)
        use_observation_experts = hero_cfg.get("use_observation_experts", True)
        use_integration_layer = hero_cfg.get("use_integration_layer", True)
        
        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            llama_model=llama_model,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            low_resource=low_resource,
            end_sym=end_sym,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            chat_template=chat_template,
            use_grad_checkpoint_llm=use_grad_checkpoint_llm,
            max_context_len=max_context_len,
            visual_dim=visual_dim,
            video_dim=video_dim,
            audio_dim=audio_dim,
            au_dim=au_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_qformer_layers=num_qformer_layers,
            num_output_queries=num_output_queries,
            include_synergy=include_synergy,
            use_observation_experts=use_observation_experts,
            use_integration_layer=use_integration_layer,
            lambda_stmil=hero_cfg.get("lambda_stmil", 0.1),
            lambda_sccl=hero_cfg.get("lambda_sccl", 0.1),
        )
        
        ckpt_path = cfg.get("ckpt", "")
        if ckpt_path:
            print(f"Load HERO Checkpoint: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
            print(f"Loaded with message: {msg}")
            
        return model
