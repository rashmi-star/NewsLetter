# Install necessary libraries
!pip install transformers diffusers torch requests beautifulsoup4 secure-smtplib

# Import necessary libraries
from transformers import pipeline
from diffusers import StableDiffusionPipeline
import torch
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import requests
from bs4 import BeautifulSoup

# Scrape data from Wikipedia to generate the newsletter content (optional)
def collect_newsletter_data(url, max_paragraphs=2):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    paragraphs = soup.find_all('p')[:max_paragraphs]
    cleaned_content = " ".join([p.get_text() for p in paragraphs])
    return cleaned_content

# Truncate input text to avoid exceeding token length (reduce to 500 tokens)
def truncate_input_text(text, max_tokens=500):
    return " ".join(text.split()[:max_tokens])

# Generate text using GPT-2 (open-source)
def generate_text(prompt, model_name="gpt2", max_length=100):
    generator = pipeline("text-generation", model=model_name)
    generated_text = generator(prompt, max_length=max_length, num_return_sequences=1)
    return generated_text[0]['generated_text']

# Generate an image using Stable Diffusion (on CPU)
def generate_image(prompt):
    model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to("cpu")  # Use CPU instead of CUDA
    image = model(prompt).images[0]  # Generate the image
    image.save("generated_image.png")  # Save the image
    return "generated_image.png"

# Send the newsletter (text + image) via email using an HTML template
def send_newsletter(subject, body_text, image_path, to_email, from_email, password):
    # Create the email (HTML + image)
    msg = MIMEMultipart('related')
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Define HTML structure
    html_body = f"""
    <html>
    <body style="font-family: Arial, sans-serif; background-color: #f7f7f7; padding: 20px;">
        <div style="background-color: white; padding: 20px; border-radius: 10px;">
            <h2 style="color: #ff9900;">AI Generated Newsletter</h2>
            <p>{body_text}</p>
            <hr>
            <h3>Generated Image Section</h3>
            <img src="cid:image1" alt="Generated Image" width="600">
            <p>Here is an AI-generated image for your newsletter.</p>
            <hr>
            <p style="color: #888;">This is an automatically generated newsletter using AI technologies.</p>
        </div>
    </body>
    </html>
    """
    msg.attach(MIMEText(html_body, 'html'))

    # Attach the generated image
    with open(image_path, 'rb') as img_file:
        img_data = img_file.read()
    image_part = MIMEImage(img_data)
    image_part.add_header('Content-ID', '<image1>')  # Attach image with content ID
    msg.attach(image_part)

    # Send the email using Gmail's SMTP server
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()
    print(f"Email sent to {to_email}")

# Full workflow to generate text and images, and send via email with template
def generate_newsletter_and_send(text_prompt, image_prompt, email_subject, to_email, from_email, email_password):
    # Generate text for the newsletter
    generated_text = generate_text(text_prompt)
    print("Generated Text:\n", generated_text)

    # Generate an image
    generated_image_path = generate_image(image_prompt)

    # Send email with the generated content
    send_newsletter(email_subject, generated_text, generated_image_path, to_email, from_email, email_password)

# Example usage
def main():
    # Email details
    email_subject = "AI Generated Newsletter: A Glimpse into the Future"
    to_email = ""  # The recipient's email
    from_email = ""  # Your gmail
    email_password = ""  # Gmail App Password

    # Text prompt for GPT-2
    text_prompt = "Generate a creative introduction about futuristic technology and its impact on humanity."

    # Image prompt for Stable Diffusion
    image_prompt = "A futuristic city skyline with flying cars, AI-powered buildings, and a clear sky etc."

    # Run the full process
    generate_newsletter_and_send(text_prompt, image_prompt, email_subject, to_email, from_email, email_password)

# Run the main function
if __name__ == '__main__':
    main()
