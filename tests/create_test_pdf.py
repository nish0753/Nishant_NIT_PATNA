from PIL import Image, ImageDraw, ImageFont
import io

def create_pdf():
    images = []
    for i in range(3):
        img = Image.new('RGB', (500, 500), color='white')
        d = ImageDraw.Draw(img)
        d.text((10, 10), f"This is Page {i+1} of the bill.", fill='black')
        d.text((10, 50), f"Item {i+1}: $100", fill='black')
        images.append(img)
    
    pdf_path = "test_multipage.pdf"
    images[0].save(
        pdf_path, "PDF", resolution=100.0, save_all=True, append_images=images[1:]
    )
    print(f"Created {pdf_path}")

if __name__ == "__main__":
    create_pdf()
