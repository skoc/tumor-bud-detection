from utils.utils import *

def get_iou(gt, pr, n_classes, EPS=1e-12):
    
    class_wise = np.zeros(n_classes)

    for cl in range(n_classes):

        intersection = np.sum((gt == cl)*(pr == cl))
        union = np.sum(np.maximum((gt == cl), (pr == cl)))
        iou = float(intersection)/(union + EPS)
        class_wise[cl] = iou

    return class_wise

def write_iou_per_bud(img_write, img_ground, img_pred, thold_area=100):
    
    # debug
    dict_locs = list()
    
    conts_ground, hierachy = cv2.findContours(img_ground, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for j, cont in enumerate(conts_ground):

        x,y,w,h = cv2.boundingRect(cont)

        if w*h > thold_area:
            
            # Bud
            bud_crop_ground = img_ground[y:y+h, x:x+w]
            bud_crop_pred = img_pred[y:y+h, x:x+w]
            
            # Calculate IoU
            score_iou = get_iou(bud_crop_ground, bud_crop_pred, n_classes=1)[0]
            # font
            font = cv2.FONT_HERSHEY_SIMPLEX
            # fontScale
            fontScale = 1
            # Blue color in BGR
            color = (0, 0, 255)
            # Line thickness of 2 px
            thickness = 2

            # Using cv2.putText() method
            image = cv2.putText(img_write, ""+str(round(score_iou,2)), (x,y), font, 
                               fontScale, color, thickness, cv2.LINE_AA)
            dict_locs.append(score_iou)
    
    return dict_locs

def generate_visuals(dir_img, dir_pred, img_count=1, clean=True, thold_iou=0.5, img_size=512):

    # list all files in dir
    files = [f for f in os.listdir(os.path.join(dir_img, 'img'))]

    # select 0.1 of the files randomly 
    random_files = np.random.choice(files, img_count)
    
    # Write Generated Visualization
    dir_write = 'visualization'
    if not os.path.exists(dir_write):
        os.makedirs(dir_write)
    
    for i, file in enumerate(random_files):
    
        sample_img = os.path.join(dir_img, 'img', file)
        file_name = ('-'.join(file.split('-')[1:])).split('.')[0]
        
        print(file)
        sample_ann = os.path.join(dir_img, 'ann', 'ann-'+file_name+'.jpg')
        sample_pred = os.path.join(dir_pred, 'pred-'+file)
        sample_mask = os.path.join(dir_img, 'mask', 'bw-'+file_name+'.png')
        
        if not os.path.exists(sample_ann): continue
        fig, axes = plt.subplots(1, 4, figsize=(72, 72))
        ax = axes.flatten()

        orj_img = cv2.imread(sample_img,cv2.IMREAD_COLOR)
        ann_img = cv2.imread(sample_ann, cv2.IMREAD_COLOR)
        mask_img  = cv2.imread(sample_mask, cv2.IMREAD_GRAYSCALE)
        pred_img = cv2.imread(sample_pred, cv2.IMREAD_GRAYSCALE)
        overlap_img = mapper_image(img_ann=read_image(sample_ann, img_size=img_size), img_pred=read_image(sample_pred, img_size=img_size, mask=True),\
                                  fname="overlap-"+file, output_dir='.', clean=clean)

        # iou_scores = write_iou_per_bud(overlap_img, read_image(sample_mask, img_size=img_size, mask=True), read_image(sample_pred, img_size=img_size, mask=True), thold_area=100)
        # print(f"Score: {sum([i > thold_iou for i in iou_scores])/len(iou_scores)}")

        # select only masked area below
        # masked = input_img.copy()
        # masked[mask_img == 0 ] = 0
        if os.path.exists(sample_ann):
            
            ax[0].imshow(ann_img)
            ax[0].set_axis_off()
            ax[0].set_title("Ann Image", fontsize=60)

            ax[2].imshow(mask_img, cmap="gray")
            ax[2].set_axis_off()
            ax[2].set_title("Mask", fontsize=60)

            ax[1].imshow(overlap_img)
            ax[1].set_axis_off()
            ax[1].set_title("Overlap Image", fontsize=60)

            ax[3].imshow(pred_img, cmap="gray")
            ax[3].set_axis_off()
            ax[3].set_title("Predicted", fontsize=60)
            
            plt.savefig(os.path.join(dir_write, 'visual-generated-'+file))