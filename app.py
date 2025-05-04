import gradio as gr
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16
)

pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    safety_checker=None
).to("cuda")

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def generate(image, prompt="a person posing"):
    result = pipe(prompt=prompt, image=image, num_inference_steps=20).images[0]
    return result

demo = gr.Interface(fn=generate, inputs=[gr.Image(type="pil"), gr.Textbox()], outputs="image")
demo.launch()
