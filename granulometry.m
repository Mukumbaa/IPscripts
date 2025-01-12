function granulometry(f,seType,maxsize)
    % f = imread(img);
    subplot(2,3,1), imshow(f)
    t = 128;
    a = zeros(maxsize,1); %vettore contenente le aree di ogni iterazione
    for i=1:maxsize
        %si creano elementi strutturanti sempre più grandi e si salva ogni volta l'area nel vettore
        se = strel(seType,i);
        op = imopen(f,se);
        bw = op>=t;
        a(i) = sum(bw(:)); %(l'area è data dalla somma dei pixel che rispettano la soglia)
    end
    
    subplot(2,3,2), plot(a); %(grafico andamento area)
    subplot(2,3,3), plot(abs(diff(a))); %(grafico andamento differenze)
    se = strel(seType,3);
    smooth = imclose(imopen(f,se),se);
    subplot(2,3,4), imshow(smooth) %(facendo la stessa cosa sull'immagine smooth si avranno risultati diversi)
    
    for i=1:maxsize
        se = strel(seType,i);
        op = imopen(smooth,se);
        bw = op>=t;
        a(i) = sum(bw(:));
    end
    
    subplot(2,3,5), plot(a);
    subplot(2,3,6), plot(abs(diff(a)));
end