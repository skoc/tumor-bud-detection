def data_generator(img_files, mask_files, batch_size, IMG_HEIGHT=2048, IMG_WIDTH=2048, IMG_CHANNELS=3):

    path = img_files + "img/"
    pathMask = mask_files + "mask/"

    total = int(sum([len(files) for r, d, files in os.walk(path)]))

    _, _, filesInPath = next(os.walk(path))
    _, _, filesInPathMask = next(os.walk(pathMask))
#     print(f'total{total}')
    print(f'# of Samples Image: {len(filesInPath)}\t# of samples mask: {len(filesInPathMask)}')

    filesInPath = sorted(filesInPath)
    filesInPathMask = sorted(filesInPathMask)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < total:
            limit = min(batch_end, total)
            X_train = np.ndarray((batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)
            Y_train = np.ndarray((batch_size, IMG_HEIGHT, IMG_WIDTH, 1), dtype=np.float32)

            for i, f in enumerate(filesInPath[batch_start:limit]):
                img = cv2.imread(path + f)
                img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                img = img / 255
                X_train[i] = img

            for i, fm in enumerate(filesInPathMask[batch_start:limit]):
                img_mask = cv2.imread(pathMask + fm, cv2.IMREAD_GRAYSCALE)
                img_mask = cv2.resize(img_mask, (IMG_HEIGHT, IMG_WIDTH), interpolation=cv2.INTER_AREA)
                img_mask = img_mask / 255
                img_mask = np.expand_dims(img_mask, axis=-1)
                Y_train[i] = img_mask

            if batch_start == 0:
                print(f"[DEBUG][INFO] Data Matrix: {round(X_train.nbytes / (1024 * 1000.0), 3)} mb")

            pixels = Y_train.flatten().reshape(len(Y_train), IMG_HEIGHT * IMG_WIDTH)
            pixels = np.expand_dims(pixels, axis = -1)
            yield (X_train, pixels)
            
            batch_start += batch_size
            batch_end += batch_size