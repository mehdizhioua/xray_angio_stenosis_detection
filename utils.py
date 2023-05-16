
import torch

def preprocess(input_try,box_try):

    #reshape image input 
    input_try = input_try.permute(0, 3, 1, 2) # shape (batch_size,3,x_dim,y_dim)
    input_try = input_try.float() / 255.0

    #reshape box input
    boxes_flat = [item for sublist in box_try for item in sublist]
    boxes_stack = torch.stack(boxes_flat)
    box_try = boxes_stack.unsqueeze(0).permute(2, 0, 1) #has shape (batch_size,number_of_boxes,4)
    labels = torch.ones((2, 1)).long()

    images = list(image for image in input_try)
    targets=[]
    for i in range(len(images)):
        d = {}
        d['boxes'] = box_try[i]
        d['labels'] = labels[i]
        targets.append(d)

    return images,targets

