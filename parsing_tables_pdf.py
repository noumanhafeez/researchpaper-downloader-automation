import os
import requests
url = "https://store.veritaspress.com/site/assets/000775-secret-garden-comp-guide-lis.pdf"
local_file = "sutter_barto.pdf"
with requests.get(url, stream=True) as response:
    response.raise_for_status()
    with open(local_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)



pipeline_options = PdfPipelineOptions()
pipeline_options.generate_picture_images = True
res = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
    }
).convert(local_file)
doc = res.document