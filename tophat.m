function topHatImage = tophat(image, seType, seSize)
    if size(image, 3) == 3
        error('L''immagine deve essere in scala di grigi.');
    end
    % Crea l'elemento strutturante
    se = strel(seType, seSize);
    % Applica l'apertura morfologica
    imgo = imopen(image, se);
    % Calcola la trasformazione Top-Hat
    topHatImage = image - imgo;
end