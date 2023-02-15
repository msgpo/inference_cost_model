# Cost Estimation
### Setting
- Model: gpt-j-6b (model dim: 4096, num of layers: 28)
- Batch size: 16
- Input sequence length: 512
- Output sequence length: 256

| GPU-type  | TFLOPS  | HBM | dRAM  | tp degree | total-time (s) | throughput (token/s) |
|----|----|----|----|----|----|----|
|A100-80G|312|1935|80|1|3.677|1113.887|
|RTX-Ada-6000-48G|364|960|48|1|6.383|641.697|
|A100-40G|312|1555|40|1|4.359|939.707|
|RTX-4090|380|1000|24|1|6.126|668.609|
|A6000|149.7|768|48|1|8.878|461.353|
|A40|149.7|696|48|1|9.605|426.436|
|A30|165|933|24|1|7.464|548.781|
|V100-32GB|125|900|32|1|8.213|498.704|
|RTX-3090-Ti|142.5|1008|24|1|7.299|561.206|
|RTX-3090|142.5|935|24|1|7.717|530.804|
|RTX-Quadro-6000|130.5|672|24|1|10.154|403.370|
|RTX-Titan|130.5|672|24|1|10.154|403.370|
|A10|125|600|24|1|11.212|365.335|
|A5000|91|768|24|1|10.072|406.666|
|V100-16GB|125|900|16|1|8.213|498.704|
|RTX-3080-Ti|136.4|912|12|2|24.975|164.007|
|A4500|94.5|640|20|1|11.365|360.410|
|RTX-3080-12G|122|912|12|2|25.094|163.223|
|RTX-3060-Ti-10G|129.6|448|12|2|28.092|145.805|
|A4000|76.7|448|16|1|15.659|261.569|
|T4|65.2|300|16|1|22.240|184.176|
|GTX-Titan-V|110.0|651|12|2|26.405|155.124|
|RTX-3060-12G|101.9|360|12|2|29.855|137.195|
|RTX-2080-Ti|107.6|616|11|2|26.668|153.591|
|RTX-3060-Ti-8G|129.6|608|8|2|26.507|154.524|
|RTX-2080-Super|89.2|496|8|2|27.994|146.319|
|RTX-3070-Ti|87|608|8|2|27.031|151.531|
|RTX-3070|81|448|8|2|28.734|142.550|
|RTX-2080|80.5|448|8|2|28.744|142.497|
|RTX-3050|72.8|224|8|2|34.950|117.196|
|RTX-2070-Super|72.5|448|8|2|28.934|141.562|
|RTX-2070|59.7|448|8|2|29.344|139.586|
|RTX-2060-Super|57.4|448|8|2|29.437|139.145|
|RTX-2060|41.9|336|6|4|37.169|110.200|