import io
import os
import warnings

import replicate
from stability_sdk import client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
import io, os
from numpy import random
from PIL import Image, ImageDraw, Image, ImageFont
import pandas as pd
import random
import string
import webbrowser
import urllib
import requests
from base64 import b64encode
import string
import metaseg


def set_engine(engine:str):
# Set up our connection to the API.
  stability_api = client.StabilityInference(
      key=os.environ['STABILITY_KEY'], # API Key reference.
      verbose=True, # Print debug messages.
      engine = engine, # Set the engine to use for generation. For SD 2.0 use "stable-diffusion-v2-0".
      # Available engines: stable-diffusion-v1 stable-diffusion-v1-5 stable-diffusion-512-v2-0 stable-diffusion-768-v2-0 stable-inpainting-v1-0 stable-inpainting-512-v2-0
  )
  return stability_api


sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"
device = "cuda"


sam = metaseg.sam_model_registry[model_type](checkpoint='sam_vit_h_4b8939.pth')
predictor = metaseg.SamPredictor(sam)
sam.to(device="cpu")

#@title
import numpy as np
import openai
import cv2
import gradio as gr
from serpapi import GoogleSearch
import random
import replicate
import time

openai.api_key = os.environ['OPENAI_API_KEY']


def generate_prompt(body_type,exterior_color,roof_type,wheels_and_tires):
  response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Generate description of automobile from the following desciptors-Realistic image of an automobile with {exterior_color} color {body_type} with {roof_type}.Start with word draw do not start with this.Do not describe about interior of the car.",
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
  result = response["choices"][0]["text"]
  print(result)
  return result


def text_to_image(prompt,guidance_model,cfg,engine):

  # Set up our initial generation parameters.
  print(f'text_to_image: {prompt}')
  stability_api = set_engine(engine)
  answers = stability_api.generate(
      prompt=prompt,
      seed=992446758, # If a seed is provided, the resulting generated image will be deterministic.
                      # What this means is that as long as all generation parameters remain the same, you can always recall the same image simply by generating it again.
                      # Note: This isn't quite the case for CLIP Guided generations, which we tackle in the CLIP Guidance documentation.
      steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=cfg, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
      width=512, # Generation width, defaults to 512 if not included.
      height=512, # Generation height, defaults to 512 if not included.
      samples=1, # Number of images to generate, defaults to 1 if not included.
      sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL ,# Choose which sampler we want to denoise our generation with.
                                                  # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                  # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
      guidance_preset = generation.GUIDANCE_PRESET_FAST_GREEN,
      # guidance_strength = guidance_strength,
      guidance_models = [guidance_model]
  )

  # Set up our warning to print to the console if the adult content classifier is tripped.
  # If adult content classifier is not tripped, save generated images.
  for resp in answers:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn(
                  "Your request activated the API's safety filters and could not be processed."
                  "Please modify the prompt and try again.")
          if artifact.type == generation.ARTIFACT_IMAGE:
              img = Image.open(io.BytesIO(artifact.binary))
              # img.save(str(artifact.seed)+ ".png") # Save our generated images with their seed number as the filename.
  return img


def imag2img(prompt,init_image,guidance_model,cfg,engine):
  stability_api = set_engine(engine)
  # Set up our initial generation parameters.
  # init_image = init_image.resize((512,512))
  answers2 = stability_api.generate(
      prompt=prompt,
      init_image=init_image, # Assign our previously generated img as our Initial Image for transformation.
      start_schedule=0.6, # Set the strength of our prompt in relation to our initial image.
      seed=992446758, # If attempting to transform an image that was previously generated with our API,
                      # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
      steps=30, # Amount of inference steps performed on image generation. Defaults to 30.
      cfg_scale=cfg, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
      width=512, # Generation width, defaults to 512 if not included.
      height=512, # Generation height, defaults to 512 if not included.
     sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL ,# Choose which sampler we want to denoise our generation with.
                                                  # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                  # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
      guidance_preset = generation.GUIDANCE_PRESET_FAST_GREEN,
      # guidance_strength = guidance_strength,
      guidance_models = [guidance_model]
  )

  # Set up our warning to print to the console if the adult content classifier is tripped.
  # If adult content classifier is not tripped, save generated image.
  for resp in answers2:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn(
                  "Your request activated the API's safety filters and could not be processed."
                  "Please modify the prompt and try again.")
          if artifact.type == generation.ARTIFACT_IMAGE:
              
              img2 = Image.open(io.BytesIO(artifact.binary))
  return img2
def inpainting(prompt,image,mask,guidance_model,cfg,engine):
  stability_api = set_engine(engine)
  # image = Image.fromarray(image).resize((512,512))
  # mask = Image.fromarray(mask).resize((512,512))
  # init_image = image['image'].convert("RGB").resize((512,512))
  # mask = np.array(image['mask'].convert("RGB").resize((512,512)))
  # print(mask.__class__)
  # mask[mask==255] = 125
  # mask[mask==0] = 255
  # mask[mask==125] = 0
  # mask = Image.fromarray(mask)
  answers = stability_api.generate(
      prompt=prompt,
      init_image=Image.fromarray(image), # Assign our previously generated img as our Initial Image for transformation.
      mask_image=Image.fromarray(mask),
      start_schedule=1, # Set the strength of our prompt in relation to our initial image.
      seed=992446758, # If attempting to transform an image that was previously generated with our API,
                      # initial images benefit from having their own distinct seed rather than using the seed of the original image generation.
      steps=30, # Amount of inference steps performed on image generation. Defaults to 30. 
      cfg_scale=cfg, # Influences how strongly your generation is guided to match your prompt.
                    # Setting this value higher increases the strength in which it tries to match your prompt.
                    # Defaults to 7.0 if not specified.
      width=512, # Generation width, defaults to 512 if not included.
      height=512, # Generation height, defaults to 512 if not included.
      sampler=generation.SAMPLER_K_DPMPP_2S_ANCESTRAL ,# Choose which sampler we want to denoise our generation with.
                                                  # Defaults to k_lms if not specified. Clip Guidance only supports ancestral samplers.
                                                  # (Available Samplers: ddim, plms, k_euler, k_euler_ancestral, k_heun, k_dpm_2, k_dpm_2_ancestral, k_dpmpp_2s_ancestral, k_lms, k_dpmpp_2m)
      guidance_preset = generation.GUIDANCE_PRESET_FAST_GREEN,
      # guidance_strength = guidance_strength,
      guidance_models = [guidance_model]
  )


  # Set up our warning to print to the console if the adult content classifier is tripped. If adult content classifier is not tripped, display generated image.
  for resp in answers:
      for artifact in resp.artifacts:
          if artifact.finish_reason == generation.FILTER:
              warnings.warn(
                  "Your request activated the API's safety filters and could not be processed."
                  "Please modify the prompt and try again.")
          if artifact.type == generation.ARTIFACT_IMAGE:
              img2 = Image.open(io.BytesIO(artifact.binary)) # Set our resulting initial image generation as 'img2' to avoid overwriting our previous 'img' generation.
              
  return img2
selected_pixels = []
exception = ''

def crop_image(image,mask):
  
  
  print(image.__class__)
  Image.fromarray(image).save('/tmp/image.png')
  image = cv2.imread('/tmp/image.png')
  Image.fromarray(mask).save('/tmp/mask.png')
  mask = cv2.imread('/tmp/mask.png',0)
  
  mask[mask==0] = 125
  mask[mask==255]=0
  mask[mask==125]=255

  # Set 0 as threshold to create a binary mask
  ret, binary_mask = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY)

  # Find contours in the binary mask
  contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # Get the bounding box of the contour
  x, y, w, h = cv2.boundingRect(contours[0])

  # Cut the region of interest (ROI) from the original image
  roi = image[y:y+h, x:x+w]
  Image.fromarray(roi).save('cropped.png')

  url = upload_img_url('cropped.png')

  for i in range(1,15):
    try:
      serpai_op = use_serpapi(url,i)
      break  
    except Exception as e:
      print(e)

  return serpai_op

def use_serpapi(image_url,number=0):
  
  api_key = os.environ[f'SERPAPI_KEY{number}']
  params = {
    "engine": "google_reverse_image",
    "image_url": image_url,
    "api_key": api_key
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  print(f'Results SERPAPI:{results}')
  inline_images = results["inline_images"][:5]
  
    

  
  shopping_results = []
  shopping_results+=[result_dct for result in inline_images for result_dct in shopping_url(result['title'],api_key)]
  car_df = pd.DataFrame(shopping_results)
  
  return car_df
  

def shopping_url(product_title,api_key):
  params = {
    "engine": "google_shopping",
    "q": product_title,
    "api_key": api_key
  }

  search = GoogleSearch(params)
  results = search.get_dict()
  shopping_results = results["shopping_results"]
  return [{'title': result_dict['title'],
                      'price':result_dict['price'], 'link': result_dict['link'],
                      } 
                     for result_dict in shopping_results ]


def upload_img_url(image_path):
    API_ENDPOINT = "https://api.imgbb.com/1/upload"
    

    with open(image_path, "rb") as image:
        image_data_ = b64encode(image.read())
        image_data = image_data_.decode('utf-8')
    
    payload = {
        "key": os.environ['IMGBB_API_KEY'],
        "image": image_data
    }
    # Send the API request
    response = requests.post(API_ENDPOINT, payload)
    print(response)
    # Get the generated link from the API response
    response_json = response.json() # 
    print("Response json:", response_json)
    image_url = response_json["data"]["url"]
    print("Generated link:", image_url)
    return image_url

def generate_mask(image,evt:gr.SelectData):
  t1 = time.perf_counter()
  print(f"image inpainting: {image}")
  print('ENter fucntion')
  
  selected_pixels.append(evt.index)
  print(f'Selected {selected_pixels}')
  
  predictor.set_image(image)
  
  print('After predictor.set_image(')
  input_points = np.array(selected_pixels)
  input_labels = np.ones(input_points.shape[0])
  print('After labels')
  mask, _, _ = predictor.predict(
      point_coords = input_points, 
      point_labels = input_labels,
      multimask_output = False
  )
  print(f'mask: {mask}')
  mask = np.logical_not(mask)
  mask = Image.fromarray(mask[0,:,:])
  print(f'RETURNING TYPE{mask.__class__}')
  NUM = random.randint(1,100000000000000000000000)
  print(f"mask_{NUM}.png")
  print(f'{time.perf_counter()-t1} seconds')
  
  return mask


def depth2image_replicate(image,positive_prompt,negative_prompt,number_of_inference_steps,scheduler):
  image.save("/tmp/uploaded_image_depth_to_image.png")
  response = replicate.run(
     "jagilley/stable-diffusion-depth2img:68f699d395bc7c17008283a7cef6d92edc832d8dc59eb41a6cafec7fc70b85bc",
     input = {
         "prompt": positive_prompt,
         
         "input_image": open("/tmp/uploaded_image_depth_to_image.png","rb"),
         "num_inference_steps":int(number_of_inference_steps),
         "negative_prompt": negative_prompt,
         "scheduler": scheduler

         
     })
  resp = requests.get(response[0])
  with open('/tmp/file.png','wb') as file:
    file.write(resp.content)

  return Image.open(file.name)
  
  
def inpainting_replicate(prompt,img,mask,number_of_inference_steps):
  Image.fromarray(img).save('/tmp/inpaining_input.png')
  Image.fromarray(mask).save('/tmp/inpaining_mask.png')
  output = replicate.run(
    "andreasjansson/stable-diffusion-inpainting:e490d072a34a94a11e9711ed5a6ba621c3fab884eda1665d9d3a282d65a21180",
    input={"prompt": "a futristic green luxury car with big wheels",
           "image":open('/tmp/inpaining_input.png',"rb"),
           "mask": open('/tmp/inpaining_mask.png','rb'),
           "invert_mask": True,
            "num_inference_steps":int(number_of_inference_steps),
           })
  resp = requests.get(output[0])
  with open('/tmp/inpainting_file.png','wb') as file:
    file.write(resp.content)

  return Image.open(file.name)
  

#############GRADIO INTERFACE#################################################################################
# import gradio as gr
# import gradio as gr


def prompt_use_for_next(text_prompt,*args):
  return text_prompt,text_to_image(text_prompt,*args)

def img2img_use_for_next(*args):
  return args[1],imag2img(*args) 

def inpainting_use_for_next(image):
  return image




GUIDANCE_MODELS = ['ViT-L-14--openai', 'ViT-H-14--good', 'ViT-B-32--laion2b_e16', 'ViT-L-14--laion400m_e32', 'ViT-B-32--openai', 'ViT-B-16--openai']
ENGINE_MODELS = ['stable-diffusion-xl-beta-v2-2-2','stable-diffusion-768-v2-1','stable-diffusion-512-v2-0','stable-diffusion-768-v2-0','stable-diffusion-v1']
with gr.Blocks(theme="Ajaxon6255/Emerald_Isle") as demo: 
  gr.Markdown("""<h1 style="color:white;font-family:monospace;text-align:center">TechnoForge Automotive</h1>""")
  gr.Markdown("""<p style='color:white;font-family:monospace'>Attention car enthusiasts! Are you looking for a car that combines traditional craftsmanship with cutting-edge technology? Look no further than TechnoForge Automotive! Inspired by the master blacksmith and craftsman Hephaestus, our team combines the latest in stable diffusion technology with the power of GPT-3 to create the most innovative and precise car designs on the market.Just like Hephaestus forged his creations with the utmost care and precision, we use stable diffusion to ensure the highest level of quality in every detail of our designs. And with the power of GPT-3 technology at our fingertips, we can push the limits of innovation and take car designing to new heights.</p>""")
  gr.HTML(value="<img id='HeadImage' src='https://i.ibb.co/RYqqt4Z/op9.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
  gr.HTML(value="<style>#HeadImage:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")
  ######################################################################################################################################
  with gr.Accordion(label="Generte Prompt"):
    with gr.Row():
      with gr.Column():
        body_type = gr.Textbox(label = "Type of car (eg: SUV, Sedan)")
        color = gr.Textbox(label="Car color")
        roof_type = gr.Textbox(label="Roof Type (eg: Sunroof, foldable roof)")
      with gr.Column():
        gen_prompt = gr.Textbox(label="Use this Prompt")
    gr.HTML(value="<img id='generate_prompt' src='https://i.ibb.co/XsHCsK7/1.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
    gr.HTML(value="<style>#generate_prompt:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")  
    with gr.Row(): 
      with gr.Column():
        prompt_gen = gr.Button("Generate Prompt",elem_id="gradio_button")
      with gr.Column():
        prompt_next = gr.Button("Use this text as next",elem_id="gradio_button")
    prompt_gen.click(generate_prompt,[body_type,color,roof_type],[gen_prompt])
    
    #########################################################################################################################
  with gr.Accordion(label="Text2Image"):
    with gr.Row():
      with gr.Column():
        text2imge_prompt = gr.Textbox(label="Prompt for Text")
        text2imge_guidnace_model = gr.Dropdown(value='ViT-L-14--laion400m_e32',choices=GUIDANCE_MODELS,label="Guidance Model")
        
      with gr.Column():
        text2imge_op = gr.Image(type="pil")
        text2image_cfg = gr.Slider(label="Cfg Scale",mimimum=0,maximum=8,value=7.0)
        text2imge_model = gr.Dropdown(label="Engine",choices=ENGINE_MODELS,value='stable-diffusion-xl-beta-v2-2-2')
        # op2 = gr.Image()
    gr.HTML(value="<img id='textImage' src='https://i.ibb.co/6wDWhvd/2.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
    gr.HTML(value="<style>#textImage:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")  
    with gr.Row(): 
      with gr.Column():
        text2imge_button = gr.Button("Transform",elem_id="gradio_button")
      with gr.Column():
        text2imge_next = gr.Button("Use this Image as next",elem_id="gradio_button")
    prompt_gen.click(generate_prompt,[body_type,color,roof_type],[gen_prompt])
    
    
    text2imge_button.click(text_to_image,[text2imge_prompt,text2imge_guidnace_model,text2image_cfg,text2imge_model],[text2imge_op])
    prompt_next.click(prompt_use_for_next,[gen_prompt,text2imge_guidnace_model,text2image_cfg,text2imge_model],[text2imge_prompt,text2imge_op])
    #########################################################################################################################
  # with gr.Accordion(label="Image2Image"):
  #   with gr.Row():
  #     with gr.Column():
  #       img2imge_prompt = gr.Textbox(label="Prompt for img2img")
  #       img2image_image = gr.Image(type="pil")
        
  #       img2image_guidnace_model = gr.Dropdown(value='ViT-L-14--laion400m_e32',choices=GUIDANCE_MODELS,label="Guidance Model")
  #     with gr.Column():
  #       img2image_op = gr.Image()
  #       img2img_cfg = gr.Slider(label="Cfg Scale",mimimum=0,maximum=8,value=7.0)
  #       img2img_model = gr.Dropdown(label="Engine",choices=ENGINE_MODELS,value='stable-diffusion-xl-beta-v2-2-2')
  #   with gr.Row(): 
  #     with gr.Column():
  #       img2img_button = gr.Button("Transform",elem_id="gradio_button")
  #     with gr.Column():
  #       img2img_next = gr.Button("Use this Image as next",elem_id="gradio_button")
    
    
  #   img2img_button.click(imag2img,[img2imge_prompt,img2image_image,img2image_guidnace_model,img2img_cfg,img2img_model],[img2image_op])
  #   text2imge_next.click(img2img_use_for_next,[img2imge_prompt,text2imge_op,img2image_guidnace_model,img2img_cfg,img2img_model],[img2image_image,img2image_op])
      #########################################################################################################################
  with gr.Accordion(label="Depth2Image"):
    with gr.Row():
      with gr.Column():
        depth2image_positive_prompt = gr.Textbox(label="Positive Prompt")
        depth2image_negative_prompt= gr.Textbox(label="Negative Prompt")

        depth2image_image = gr.Image(type="pil")
        
        
      with gr.Column():
        depth2image_op = gr.Image()#type="pil")
        depth2image_inference_steps = gr.Slider(label="Inference Steps",mimimum=1,maximum=500,value=50,step=1)#204
        depth2image_scheduler = gr.Dropdown(label="Scheduler",choices=['DDIM','K_EULER','DPMSolverMultistep','K_EULER_ANCESTRAL','PNDM','KLMS'],value='DPMSolverMultistep')
    gr.HTML(value="<img id='depth_to_image' src='https://i.ibb.co/c2pKpLJ/depth2image.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
    gr.HTML(value="<style>#depth_to_image:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")  
    with gr.Row(): 
      with gr.Column():
        depth2image_button = gr.Button("Transform",elem_id="gradio_button")
      with gr.Column():
        depth2image_nxt_button = gr.Button("Use Image as next")  
      # with gr.Column():
      #   img2img_next = gr.Button("Use this Image as next",elem_id="gradio_button")
     #depth2image_replicate(image,positive_prompt,negative_prompt,number_of_inference_steps,scheduler)
    
    depth2image_button.click(depth2image_replicate,[depth2image_image,depth2image_positive_prompt,depth2image_negative_prompt,
                                                    depth2image_inference_steps,depth2image_scheduler],[depth2image_op])
    text2imge_next.click(inpainting_use_for_next,text2imge_op,depth2image_image)
    #########################################################################################################################
  with gr.Accordion("Inpainting"):
    with gr.Row():
      with gr.Column():
        inpainting_prompt = gr.Textbox(label="Prompt for inpainting")
        
        inpainting_image = gr.Image()#type="pil",tool="sketch")
        inpainting_guidnace_model = gr.Dropdown(value='ViT-L-14--laion400m_e32',choices=GUIDANCE_MODELS,label="Guidance Model")
        inpainting_model = gr.Dropdown(label="Engine",choices=ENGINE_MODELS,value='stable-diffusion-xl-beta-v2-2-2')
      with gr.Column():
        inpainting_mask = gr.Image()
        inpainting_op = gr.Image()
        inpainting_cfg = gr.Slider(label="Cfg Scale",mimimum=0,maximum=8,value=7.0)
    gr.HTML(value="<img id='inpainting' src='https://i.ibb.co/tMPPVTL/inpainting.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
    gr.HTML(value="<style>#inpainting:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>") 
    with gr.Row(): 
      with gr.Column():
        inpainting_button = gr.Button("Transform",elem_id="gradio_button")
      with gr.Column():
        inpainting_button_next = gr.Button("Use this Image as next",elem_id="gradio_button")
        
        
        # op2 = gr.Image()
    inpainting_image.select(fn=generate_mask,inputs=[inpainting_image],outputs=inpainting_mask)
    depth2image_op.select(fn=generate_mask,inputs=[depth2image_op],outputs=inpainting_mask)
    



  inpainting_button.click(fn=inpainting,inputs=[inpainting_prompt,inpainting_image,inpainting_mask,inpainting_guidnace_model,inpainting_cfg,inpainting_model ],outputs=[inpainting_op]) 
  depth2image_nxt_button.click(fn=inpainting_use_for_next,inputs=[depth2image_op],outputs=[inpainting_image]) 
  
  ##################################################################################################################################################################
  # with gr.Accordion(""):
  #   with gr.Row():
  #     with gr.Column():
  #       inpainting_prompt = gr.Textbox(label="Prompt for inpainting")
        
  #       inpainting_image = gr.Image()#type="pil",tool="sketch")
  #     with gr.Column():
  #       inpainting_mask = gr.Image()
  #       inpainting_op = gr.Image()
  #       inpainting_guidnace_model = gr.Dropdown(value='ViT-L-14--laion400m_e32',choices=GUIDANCE_MODELS)
  #       # op2 = gr.Image()
  #   inpainting_image.select(fn=generate_mask,inputs=[inpainting_image],outputs=inpainting_mask)
  #   inpainting_button = gr.Button("Transform")



  # inpainting_button.click(fn=inpainting,inputs=[inpainting_prompt,inpainting_image,inpainting_mask,inpainting_guidnace_model ],outputs=[inpainting_op]) 
  ############################
  with gr.Accordion("Image Search"):
    with gr.Row():
      with gr.Column():
        search_image = gr.Image()
      with gr.Column():
        search_image_mask = gr.Image(interactive=False)
      # with gr.Column():
      #   cropped_image = gr.Image()
    

    search_op = gr.Dataframe()
    gr.HTML(value="<img id='image_search' src='https://i.ibb.co/qpZNSkb/image-Search.png' alt='Generate knowlwdge graph' width='1200' height='300' style='border: 2px solid #fff;'/>")   
    gr.HTML(value="<style>#image_search:hover{box-shadow: 0 12px 16px 0 rgba(0,0,0,0.24),0 17px 50px 0 rgba(0,0,0,0.19);}</style>")  
    search_button  = gr.Button("Search",elem_id="gradio_button")
    
    search_image.select(fn=generate_mask,inputs=[search_image],outputs=search_image_mask)
    inpainting_op.select(fn=generate_mask,inputs=[inpainting_op],outputs=search_image_mask)


    search_button.click(fn=crop_image,inputs=[search_image,search_image_mask],outputs=[search_op])
    inpainting_button_next.click(fn=inpainting_use_for_next,inputs=[inpainting_op],outputs=[search_image])
    

      




    




demo.queue(concurrency_count=3,max_size=2)
demo.launch(debug=True)