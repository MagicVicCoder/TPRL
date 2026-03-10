import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from .base_mllm import BaseMLLM

class LLaVA(BaseMLLM):
    """
    LLaVA-1.5 model implementation (supports 7B and 13B variants).
    """
    def _load_model(self):
        print(f"Loading MLLM model: {self.model_id} (this may take a while)...")
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print("MLLM model loaded successfully.")

    def get_components_for_env(self, image, question):
        # Prepare the prompt in LLaVA format
        prompt = f"USER: <image>\n{question}\nASSISTANT:"

        try:
            inputs = self.processor(
                text=prompt,
                images=image.convert("RGB"),
                return_tensors="pt"
            )
            # Move inputs to device
            input_ids = inputs['input_ids'].to(self.device)
            pixel_values = inputs['pixel_values'].to(self.device)
        except Exception as e:
            print(f"Warning: Failed to process sample. Error: {e}")
            return None

        with torch.no_grad():
            # Get vision features from the vision tower
            vision_outputs = self.model.vision_tower(pixel_values, output_hidden_states=False)
            # Extract the last hidden state from vision tower output
            if hasattr(vision_outputs, 'last_hidden_state'):
                image_features = vision_outputs.last_hidden_state
            else:
                image_features = vision_outputs[0]
            # Shape: [batch_size, num_patches, vision_hidden_size]

            # Project vision features to LLM space using multi-modal projector
            original_visual_features = self.model.multi_modal_projector(image_features)
            # Shape: [batch_size, num_patches, hidden_dim]
            current_num_patches = original_visual_features.shape[1]

            # Get text embeddings from language model
            full_embeds = self.model.language_model.get_input_embeddings()(input_ids)
            # Shape: [batch_size, seq_len, hidden_dim]

            # Find image token positions
            image_token_id = self.model.config.image_token_index
            image_token_indices = torch.where(input_ids[0] == image_token_id)[0]

            if len(image_token_indices) == 0:
                return None

            # Split text embeddings around image tokens
            img_token_start_idx = image_token_indices[0]
            img_token_end_idx = image_token_indices[-1]

            text_embeds_part1 = full_embeds[:, :img_token_start_idx, :]  # Before image
            text_embeds_part2 = full_embeds[:, img_token_end_idx + 1:, :]  # After image

            # Create query embeddings from text (mean pooling)
            text_only_embeds = torch.cat([text_embeds_part1, text_embeds_part2], dim=1)
            query_embeddings = text_only_embeds.mean(dim=1, keepdim=True)

        return {
            "original_visual_features": original_visual_features,
            "text_embeds_part1": text_embeds_part1,
            "text_embeds_part2": text_embeds_part2,
            "query_embeddings": query_embeddings,
            "current_num_patches": current_num_patches
        }

    def generate_answer(self, final_embeddings, attention_mask, max_new_tokens=20):
        with torch.no_grad():
            output_ids = self.model.language_model.generate(
                inputs_embeds=final_embeddings,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()
