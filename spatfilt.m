function f = spatfilt(g, type, m, n, parameter)

g = im2double(g);

h = ones(m, n);

% Applicazione del filtro in base al tipo selezionato
switch type
    case 'arithmetic'

        h = ones(m,n) / (m*n);

        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "arithmetic", m, n);
            end
        else
            f = imfilter(g,h);
        end



        % h = ones(m,n) / (m*n);
        % f = imfilter(g,h);
    case 'geometric'
        h = ones(m,n);

        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "geometric", m, n);
            end
        else
            f = exp(imfilter(log(g),h,'replicate')).^(1/(m*n));
        end

        % h = ones(m,n);
        % f = imfilter(log(g),h,'replicate');
        % f = exp(f);
        % f = f .^(1/numel(h));

        % f = exp(imfilter(log(g),h,'replicate')).^(1/(m*n));
    case 'harmonic'


        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "harmonic", m, n);
            end
        else
            f = (m*n) ./ imfilter(1./(g + eps), h, 'replicate');
        end
        % f = (m*n) ./ imfilter(1./(g + eps), h, 'replicate');
    case 'contraharmonic'

        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "contraharmonic", m, n,parameter);
            end
        else
            f = imfilter(g.^(parameter+1),h,"replicate")./(imfilter(g.^(parameter),h,"replicate")+eps);
        end


        % f = imfilter(g.^(parameter+1),h,"replicate")./(imfilter(g.^(parameter),h,"replicate")+eps);
    case 'median'
        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "median", m, n);
            end
        else
            f = medfilt2(g, [m n]);
        end
    case 'min'
        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "min", m, n);
            end
        else
            f = ordfilt2(g, 1, h);
        end
        % f = ordfilt2(g, 1, h);
    case 'max'
        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "max", m, n);
            end
        else
            f = ordfilt2(g, m*n, h);
        end
        % f = ordfilt2(g, m*n, h);
    case 'midpoint'
        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "midpoint", m, n);
            end
        else
            f = (ordfilt2(g, 1, h) + ordfilt2(g, m*n, h)) / 2;
        end
        % f = (ordfilt2(g, 1, h) + ordfilt2(g, m*n, h)) / 2;
    case 'alphabalanced' 


        if size(g, 3) == 3
            f = zeros(size(g), 'like', g);

            for c = 1:3
                f(:, :, c) = spatfilt(g(:, :, c), "alphabalanced", m, n, parameter);
            end
        else
            d = parameter;
    
            f = zeros(size(g,1),size(g,2));
    
            g = padarray(g,[floor(m/2),floor(n/2)],0,"both");
    
            for i=(floor(m/2)+1):(floor(m/2)+size(f,1))
                for j=(floor(n/2)+1):(floor(n/2)+size(f,2))
                    
                    x1 = j-floor(n/2);
                    x2 = j+floor(n/2);
                    y1 = i-floor(m/2);
                    y2 = i+floor(m/2);
    
                    s = g(y1:y2,x1:x2);
    
                    s = sort(s(:));
    
                    f(i-floor(m/2),j-floor(n/2)) = mean(s(floor(d/2)+1:(end-floor(d/2)-1)));
    
    
                end
            end
        end

    otherwise
        error('Tipo di filtro non valido');
end

f = cast(f, class(g));
end