function image = morf_grad(image, seType, seSize)

    % Verifica che l'immagine sia in scala di grigi
    if size(image, 3) == 3
        error('L''immagine deve essere in scala di grigi.');
    end

    se = strel(seType,seSize);
    imaged = imdilate(image,se);
    imager = imerode(imaged,se);
    image = imaged - imager;
end