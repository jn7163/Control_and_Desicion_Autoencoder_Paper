#!/bin/bash

for i in `ls|grep dot`;do
dot -T png -o ${i%.dot}.png $i
convert ${i%.dot}.png ${i%.dot}.eps
rm ${i%.dot}.png
done

convert dropout.eps  -rotate 270 b.eps
mv b.eps dropout.eps

convert denoised_autoencoder.eps  -rotate 270 b.eps
mv b.eps denoised_autoencoder.eps

convert autoencoder_structure.eps  -rotate 270 b.eps
mv b.eps autoencoder_structure.eps
