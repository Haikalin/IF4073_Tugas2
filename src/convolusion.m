function output = convolusion(imagePath, masking)
    img = imread(imagePath);
    img = im2double(img);  % skala ke 0–1
    
    % Normalisasi kernel (jika bukan deteksi tepi)
    if sum(masking(:)) ~= 0
        masking = masking / sum(masking(:));
    end
    
    [rows, cols, channels] = size(img);
    [m, n] = size(masking);
    
    offset_r = floor(m/2);
    offset_c = floor(n/2);
    
    output = zeros(rows, cols, channels);
    
    for c = 1:channels
        for i = (1 + offset_r):(rows - offset_r)
            for j = (1 + offset_c):(cols - offset_c)
                region = img(i-offset_r:i+offset_r, j-offset_c:j+offset_c, c);
                val = sum(sum(region .* masking));
                val = max(0, min(1, val));
                output(i, j, c) = val;
            end
        end
        
        % Salin border dari original
        output(1:offset_r, :, c) = img(1:offset_r, :, c);
        output(rows-offset_r+1:rows, :, c) = img(rows-offset_r+1:rows, :, c);
        output(:, 1:offset_c, c) = img(:, 1:offset_c, c);
        output(:, cols-offset_c+1:cols, c) = img(:, cols-offset_c+1:cols, c);
    end
    
    output = im2uint8(output);  % balik ke uint8
end
