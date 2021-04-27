clear; close all; clc

A = [ 52  55  61  66  70  61  64  73;
      63  59  55  90 109  85  69  72;
      62  59  68 113 144 104  66  73;
      63  58  71 122 154 106  70  69;
      67  61  68 104 126  88  68  70;
      79  65  60  70  77  68  58  75;
      85  71  64  59  55  61  65  83;
      87  79  69  68  65  76  78  94];

B = encode(A);
Ad = decode(B);

% imwrite(uint8(A), 'A.bmp');
% imwrite(uint8(A), 'A.jpg', 'Quality', 50);

function a = alpha(u)
a = ones(size(u));
a(u == 1) = 1/sqrt(2);
end

function Ts = quantize(quality, chrominance)
if nargin == 1
    chrominance = false;
end
if ~chrominance
    Tb = [ 16  11  10  16  24  40  51  61;
           12  12  14  19  26  58  60  55;
           14  13  16  24  40  57  69  56;
           14  17  22  29  51  87  80  62;
           18  22  37  56  68 109 103  77;
           24  35  55  64  81 104 113  92;
           49  64  78  87 103 121 120 101;
           72  92  95  98 112 100 103  99];
else
    Tb = [17 18 24 47 99 99 99 99;
          18 21 26 66 99 99 99 99;
          24 26 56 99 99 99 99 99;
          47 66 99 99 99 99 99 99;
          99 99 99 99 99 99 99 99;
          99 99 99 99 99 99 99 99;
          99 99 99 99 99 99 99 99;
          99 99 99 99 99 99 99 99];
end
if (quality < 50)
    S = 5000/quality;
else
    S = 200 - 2*quality;
end
Ts = floor((S*Tb + 50) / 100);
Ts(Ts == 0) = 1;
end

function B = encode(A, quality, chrominance)
if nargin < 3
    chrominance = false;
end
if nargin < 2
    quality = 50;
end
g = A - 128;
G = zeros(8);
ridx = repmat((1:8)', 1, 8);
cidx = repmat(1:8, 8, 1);
for u=1:8
    for v=1:8
        temp = g .* cos((2*ridx-1)*(u-1)*pi/16) .* cos((2*cidx-1)*(v-1)*pi/16);
        G(u,v) = 0.25 * alpha(u) * alpha(v) * sum(temp(:));
    end
end
Q = quantize(quality, chrominance);
B = round(G ./ Q);
end

function A = decode(B, quality, chrominance)
if nargin < 3
    chrominance = false;
end
if nargin < 2
    quality = 50;
end
Q = quantize(quality, chrominance);
F = B .* Q;
f = zeros(8);
ridx = repmat((1:8)', 1, 8);
cidx = repmat(1:8, 8, 1);
for x=1:8
    for y=1:8
        temp = alpha(ridx) .* alpha(cidx) .* F .* cos((2*x-1)*(ridx-1)*pi/16) .* cos((2*y-1)*(cidx-1)*pi/16);
        f(x,y) = 0.25 * sum(temp(:));
    end
end
A = round(f) + 128;
end
