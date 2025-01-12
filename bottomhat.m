function bottomhatimg = bottomhat(image,seType,seSize)
    if size(image, 3) == 3
        error('L''immagine deve essere in scala di grigi.');
    end
    % Crea l'elemento strutturante
    se = strel(seType, seSize);
    % Applica l'apertura morfologica
    imgc = imclose(image, se);
    % Calcola la trasformazione Top-Hat
    bottomhatimg = imgc - image;
end