# Cost Estimation
### Setting
- Model: gpt-j-6b (model dim: 4096, num of layers: 28)
- Batch size: 1
- Input sequence length: 128
- Output sequence length: 32

| GPU-type  | TFLOPS  | HBM | dRAM  | tp degree | total-time (s) | throughput (token/s) |
|----|----|----|----|----|----|----|
|A100-80G|312|1935|80|1|0.370|86.556|
|RTX-Ada-6000-48G|364|960|48|1|0.732|43.729|
|A100-40G|312|1555|40|1|0.457|69.988|
|RTX-4090|380|1000|24|1|0.702|45.552|
|A6000|149.7|768|48|1|0.926|34.541|
|A40|149.7|696|48|1|1.020|31.379|
|A30|165|933|24|1|0.765|41.850|
|V100-32GB|125|900|32|1|0.799|40.057|
|RTX-3090-Ti|142.5|1008|24|1|0.713|44.892|
|RTX-3090|142.5|935|24|1|0.766|41.749|
|RTX-Quadro-6000|130.5|672|24|1|1.059|30.220|
|RTX-Titan|130.5|672|24|1|1.059|30.220|
|A10|125|600|24|1|1.184|27.030|
|A5000|91|768|24|1|0.942|33.971|
|V100-16GB|125|900|16|1|0.799|40.057|
|RTX-3080-Ti|136.4|912|12|2|0.667|48.005|
|A4500|94.5|640|20|1|1.121|28.546|
|RTX-3080-12G|122|912|12|2|0.668|47.893|
|RTX-3060-Ti-10G|129.6|448|12|2|1.061|30.166|
|A4000|76.7|448|16|1|1.594|20.076|
|T4|65.2|300|16|1|2.365|13.529|
|GTX-Titan-V|110.0|651|12|2|0.822|38.925|
|RTX-3060-12G|101.9|360|12|2|1.254|25.526|
|RTX-2080-Ti|107.6|616|11|2|0.853|37.528|
|RTX-3060-Ti-8G|129.6|608|8|2|0.857|37.328|
|RTX-2080-Super|89.2|496|8|2|0.992|32.250|
|RTX-3070-Ti|87|608|8|2|0.864|37.034|
|RTX-3070|81|448|8|2|1.069|29.930|
|RTX-2080|80.5|448|8|2|1.069|29.927|
|RTX-3050|72.8|224|8|2|1.845|17.343|
|RTX-2070-Super|72.5|448|8|2|1.072|29.858|
|RTX-2070|59.7|448|8|2|1.077|29.710|
|RTX-2060-Super|57.4|448|8|2|1.078|29.676|
|RTX-2060|41.9|336|6|4|0.947|33.780|