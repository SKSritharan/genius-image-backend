import os
import torch
from diffusers import StableDiffusionPipeline
from flask import Flask, request, jsonify, send_file
import io

app = Flask(__name__)

# Load the Stable Diffusion model
model_id = "stabilityai/stable-diffusion-2"
# model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32) #torch.float16
pipe.to("cpu") #pipe.to("cuda") 

@app.route("/")
def index():
    return jsonify({ "message": "Hello, World!" })

@app.route('/generate', methods=['POST'])
def generate_image():
    data = request.get_json()

    prompt = data.get('prompt')
    amount = data.get('amount', 1)  # Default to generating 1 image
    resolution = data.get('resolution', "512x512")

    if prompt is None:
        return jsonify({'error': 'Please provide a prompt'}), 400
    
    # Parse resolution, assuming 512x512 if invalid format
    try:
        width, height = map(int, resolution.split('x'))
    except ValueError:
        width, height = 512, 512

    # Image generation with parameters
    generator = torch.Generator("cpu").manual_seed(1024) # Example seed
    images = pipe(prompt, num_images_per_prompt=amount, height=height, width=width, generator=generator).images

    image_urls = []
    for idx, image in enumerate(images):
        filename = f"generated_image_{idx}.png"
        save_path = os.path.join('public', filename)  # Assuming a 'static' folder
        image.save(save_path) 
        image_url = f"/public/{filename}"  # Construct URL relative to web server
        image_urls.append(image_url)

    return jsonify({
        'message': f'Generated {amount} images with resolution {width}x{height}',
        'image_urls': image_urls 
    })

    # # Save (or process) the image(s)
    # for idx, image in enumerate(images):
    #     image.save(f"generated_image_{idx}.png")

    # # Adjust the response as needed 
    # return jsonify({
    #     'message': f'Generated {amount} images with resolution {width}x{height}'
    # })

def main():
    app.run(port=int(os.environ.get('PORT', 80)))

if __name__ == "__main__":
    main()