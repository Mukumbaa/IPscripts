%funzione che prende in input la distribuzione di 
%probabilità dei valori di intensità di un'immagine 
%e calcola una delle misure statistiche skewness,
%la avg_intensity, la avg_contrast, la smoothness, 
%la uniformity e l'entropia in base al parametro
%misura passato in input.

function val = compute_metric(prob, intensity, misura)
    %calcolo della media
    if strcmp(misura, 'avg_intensity')
        val = sum(intensity.*prob);

    %calcolo della varianza (contrast)
    elseif strcmp(misura, 'avg_contrast')
        m = sum(intensity.*prob);
        val = sqrt(sum(((intensity - m).^2).*prob));
    
    %calcolo della skewness (terzo momento)
    elseif strcmp(misura, 'skewness')
        m = sum(intensity.*prob);
        val = sum(((intensity - m).^3).*prob);
        val = val/((numel(intensity) - 1)^2);
    
    %calcolo della smoothness (R)
    elseif strcmp(misura, 'smoothness')
        m = sum(intensity.*prob);
        variance = sum(((intensity - m).^2).*prob);
        %normalizzo il valore della varianza 
        variance = variance/((numel(intensity) - 1)^2);
        val = 1 - 1/(1 + variance);
    
    %calcolo della uniformity
    elseif strcmp(misura, 'uniformity')
        val = sum(prob.^2);
    
    %calcolo dell'entropia
    elseif strcmp(misura, 'entropy')
        %elimino i valori di probabilità nulli per evitare log(0) che è indefinito
        prob = prob(prob ~= 0);
        val = -sum(prob.*log2(prob));
    
    else
        error('misura non valida');
    end
end

%funzione che prende un' immagine e calcola un vettore con tutte le metriche della funzione compute_metrics
function metrics = compute_all_metrics(Im)
    [counts, intensity] = imhist(Im);
    %normalizzazione dell'istogramma
    counts = counts/numel(Im);
    metrics = zeros(1,6);

    %calcolo delle metriche arrotondate a 3 cifre decimali e salvandole nel vettore metrics
    metrics(1) = round(compute_metric(counts, intensity, 'avg_intensity'), 3);
    metrics(2) = round(compute_metric(counts, intensity, 'avg_contrast'), 3);
    metrics(3) = round(compute_metric(counts, intensity, 'skewness'), 3);
    metrics(4) = round(compute_metric(counts, intensity, 'smoothness'), 3);
    metrics(5) = round(compute_metric(counts, intensity, 'uniformity'), 3);
    metrics(6) = round(compute_metric(counts, intensity, 'entropy'), 3);
end

%leggo le immagini 
Im1 = imread('superconductor.jpg');
Im2 = imread('cholesterol.jpg');
Im3 = imread('microprocessor.jpg');

%faccio i crop delle immagini
Im1 = Im1(645:840, 235:430);
Im2 = Im2(435:635, 115:310);
Im3 = Im3(35:230, 40:230);

%visualizzo le immagini in un'unica figura
figure;
subplot(1,3,1);
imshow(Im1);
title('superconductor crop');
subplot(1,3,2);
imshow(Im2);
title('cholesterol crop');
subplot(1,3,3);
imshow(Im3);
title('microprocessor crop');



%calcolo le metriche per le immagini
metrics1 = compute_all_metrics(Im1);
metrics2 = compute_all_metrics(Im2);
metrics3 = compute_all_metrics(Im3);

%creo una tabella con ogni colonna la metrica e ogni riga l'immagine
T = table(metrics1', metrics2', metrics3', 'VariableNames', {'superconductor', 'cholesterol', 'microprocessor'}, 'RowNames', {'avg_intensity', 'avg_contrast', 'third moment', 'R', 'uniformity', 'entropy'});

%traspongo la tabella per scambiare righe e colonne
T_array = table2array(T)';
%estraggo i nomi delle colonne e delle righe della tabella che ho creato
row_names = T.Properties.VariableNames;
col_names = T.Properties.RowNames;

% Ricrea la tabella invertendo le righe e le colonne
T = array2table(T_array, 'RowNames', row_names, 'VariableNames', col_names);

disp(T)
