function image = morf_smooth(image, seType, times)

    % Verifica che l'immagine sia in scala di grigi
    if size(image, 3) == 3
        error('L''immagine deve essere in scala di grigi.');
    end

    % Crea l'elemento strutturante

    for i=2:times
        se = strel(seType, i);
        % Applica l'apertura morfologica
        openedImage = imopen(image, se);
        % Applica la chiusura morfologica
        image = imclose(openedImage, se);
    end
end