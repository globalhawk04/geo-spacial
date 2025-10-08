from transformers import pipeline
import requests
from PIL import Image, ImageDraw
from transformers import AutoModelForObjectDetection
from transformers import AutoImageProcessor
import torch


url = "test01.jpg"
image = Image.open(url)

#obj_detector = pipeline("object-detection", model="/home/j/Desktop/legion/play_dict/detr-second_try/checkpoint-20")
#obj_detector(image)


resulting = []

image_processor = AutoImageProcessor.from_pretrained("/home/j/Desktop/all_images/detr-5k_2/checkpoint-57000")
model = AutoModelForObjectDetection.from_pretrained("/home/j/Desktop/all_images/detr-5k_2/checkpoint-57000")

with torch.no_grad():
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    target_sizes = torch.tensor([image.size[::-1]])
    #here are the results they are going into a dictionary 
    results = image_processor.post_process_object_detection(outputs, threshold=0.00000001, target_sizes=target_sizes)[0]
    resulting.append(results)

boxes = []
scores = []

both = []


for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]

    print(
        f"Detected {model.config.id2label[label.item()]} with confidence "
       f"{score.item()} at location {box}"
    )

    boxes.append(box)
    scores.append(score)
    boths = box , score.item()
    both.append(boths)
'''
draw = ImageDraw.Draw(image)

for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
    box = [round(i, 2) for i in box.tolist()]
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    #draw.text((x, y), model.config.id2label[label.item()], fill="white")
'''
#image.show()

#print(boxes)
#print(len(boxes))

#print(sorted(scores))
#print(len(scores))


#print(both)

#print(sorted(both, key=lambda x:x[1]))

filtered = sorted(both, key=lambda x:x[1])

print(len(filtered))
print(filtered[-1:])

x = filtered[-1:]

print(x)

best_box = []
for bbox in x:
	print(bbox[1])
	score = 100 - bbox[1]
	print(score)
	best_box.append(bbox[0])

#print(best_box)



draw = ImageDraw.Draw(image)
for box in best_box :
    x, y, x2, y2 = tuple(box)
    draw.rectangle((x, y, x2, y2), outline="red", width=1)
    #draw.text((x, y), model.config.id2label[label.item()], fill="white")

image.show()



#2.829994230069133e-07
#99.99999971700058

3.724787589476364e-08
99.99999996275213

#print(x)

#both.sort(key=lambda x:[1])
#print(both[5:])

#x = both[:20]
#print(x)

'''
for x in x:
	#print(x)
	for y in x:
		print(y)

'''

#sorted(tuples, key=lambda x: x[0])
