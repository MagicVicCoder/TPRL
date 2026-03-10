import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from .base_mllm import BaseMLLM

class Qwen2_5VL(BaseMLLM):
    """
    Qwen2.5-VL model implementation.
    """
    def _load_model(self):
        print(f"Loading MLLM model: {self.model_id} (this may take a while)...")
        # Use device_map="auto" for simpler multi-GPU handling, or "balanced" if preferred
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(self.model_id, device_map="auto")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("MLLM model loaded successfully.")

    def get_components_for_env(self, image, question):
        message = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": question}]}]

        try:
            inputs = self.processor(
                text=[self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)],
                images=[image.convert("RGB")],
                padding=True,
                return_tensors="pt"
            ).to(self.device)
        except Exception as e:
            print(f"Warning: Failed to process sample. Error: {e}")
            return None

        with torch.no_grad():
            input_ids = inputs['input_ids']

            image_features_tuple = self.model.get_image_features(pixel_values=inputs['pixel_values'], image_grid_thw=inputs['image_grid_thw'])
            original_visual_features = torch.cat(image_features_tuple, dim=0).unsqueeze(0) # Shape: [batch_size=1, num_patches, hidden_dim]
            current_num_patches = original_visual_features.shape[1]

            full_embeds = self.model.get_input_embeddings()(input_ids) # Shape: [batch_size=1, seq_len, hidden_dim]

            image_token_id = self.model.config.image_token_id
            image_token_indices = torch.where(input_ids[0] == image_token_id)[0]

            if len(image_token_indices) == 0:
                 return None

            img_token_start_idx, img_token_end_idx = image_token_indices[0], image_token_indices[-1]

            text_embeds_part1 = full_embeds[:, :img_token_start_idx, :] # Before image tokens
            text_embeds_part2 = full_embeds[:, img_token_end_idx + 1:, :] # After image tokens

            text_only_embeds = torch.cat([text_embeds_part1, text_embeds_part2], dim=1)
            query_embeddings = text_only_embeds.mean(dim=1, keepdim=True) # Mean pooling for query representation

        return {
            "original_visual_features": original_visual_features,
            "text_embeds_part1": text_embeds_part1,
            "text_embeds_part2": text_embeds_part2,
            "query_embeddings": query_embeddings,
            "current_num_patches": current_num_patches
        }

     def generate_answer(self, final_embeddings, attention_mask, max_new_tokens=20):
        with torch.no_grad():
            output_ids = self.model.generate(
                inputs_embeds=final_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens
            )
        return self.processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
