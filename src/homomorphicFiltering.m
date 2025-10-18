function [output, filterMask, filterImage] = homomorphicFiltering(imagePath, cutOff, gammaL, gammaH)
    img = imread(imagePath);
    isColor = (size(img, 3) == 3);
    
    if isColor
        [M, N, ~] = size(img);
    else
        [M, N] = size(img);
    end
    
    c = 1.0;
    filterMask = createHomomorphicFilter(M, N, cutOff, gammaL, gammaH, c);
    
    if isColor
        R = processChannelHomomorphic(double(img(:,:,1)), filterMask);
        G = processChannelHomomorphic(double(img(:,:,2)), filterMask);
        B = processChannelHomomorphic(double(img(:,:,3)), filterMask);
        output = cat(3, R, G, B);
    else
        output = processChannelHomomorphic(double(img), filterMask);
    end
    
    p1 = prctile(output(:), 2);
    p99 = prctile(output(:), 98);
    output = (output - p1) / (p99 - p1);
    output(output < 0) = 0;
    output(output > 1) = 1;
    output = output * 255;
    
    output = uint8(output);
    
    filterImage = captureHomomorphicVisualization(filterMask, cutOff, gammaL, gammaH, c);
end

function filtered = processChannelHomomorphic(channel, H)
    channel(channel < 1) = 1;
    
    z = log(channel);
    
    Z = fft2(z);
    Z_shifted = fftshift(Z);
    
    S_shifted = Z_shifted .* H;
    
    S = ifftshift(S_shifted);
    s = real(ifft2(S));
    
    g = exp(s);
    
    filtered = g;
end

function H = createHomomorphicFilter(M, N, D0, gammaL, gammaH, c)
    [u, v] = meshgrid(1:N, 1:M);
    center_u = floor(N/2) + 1;
    center_v = floor(M/2) + 1;
    
    D = sqrt((u - center_u).^2 + (v - center_v).^2);
    
    H = (gammaH - gammaL) * (1 - exp(-c * (D.^2) / (D0^2))) + gammaL;
end

function filterImage = captureHomomorphicVisualization(H, D0, gammaL, gammaH, c)
    figHandle = figure('Visible', 'off', 'Position', [0 0 800 600]);
    
    [M, N] = size(H);
    visualSize = min(200, min(M, N));
    center_v = floor(M/2) + 1;
    center_u = floor(N/2) + 1;
    half_size = floor(visualSize/2);
    
    row_start = max(1, center_v - half_size);
    row_end = min(M, center_v + half_size);
    col_start = max(1, center_u - half_size);
    col_end = min(N, center_u + half_size);
    
    H_subset = H(row_start:row_end, col_start:col_end);
    [rows, cols] = size(H_subset);
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    mesh(X, Y, H_subset);
    colormap(jet);
    colorbar;
    xlabel('u (Horizontal Frequency)');
    ylabel('v (Vertical Frequency)');
    zlabel('H(u,v)');
    titleStr = sprintf('Homomorphic Filter (D0=%d) γL=%.2f γH=%.2f', D0, gammaL, gammaH);
    title(titleStr);
    view(-37.5, 30);
    grid on;
    lighting gouraud;
    shading interp;
    
    frame = getframe(figHandle);
    filterImage = frame.cdata;
    close(figHandle);
end