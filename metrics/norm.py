'''
This script normalizes various image metric scores (PSNR, SSIM, Variance, etc.) to the range [0, 1].
'''

def saturation_norm(saturation_score):
    '''
    saturation_score: The saturation score of the image
    Range: 0 to 255, where:
        0 represents achromatic (gray or white), i.e., lowest saturation.
        255 represents full saturation, i.e., maximum color intensity.
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=2*x%2F(2*x%2B10)%2Cx%2F(x%2B10)%2C0.5*x%2F(0.5*x%2B10)%2C0.3*x%2F(0.3*x%2B10)%2C0.1*x%2F(0.1*x%2B10)&xmin=0&xmax=300&ymin=0&ymax=1&var=x
    Chosen function: 0.3*x/(0.3*x+10)
    '''
    saturation_score_normed = 0.3 * saturation_score / (0.3 * saturation_score + 10)
    return saturation_score_normed

def PSNR_norm(PSNR_score):
    '''
    PSNR_score: The PSNR (Peak Signal-to-Noise Ratio) score of the image
    Range: Typically, PSNR values range from 0 to infinity, where:
        0 represents the worst image quality, indicating maximum noise.
        Higher PSNR values represent better image quality, with higher values indicating minimal noise and better similarity to the reference image.
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=3*x%2F(3*x%2B10)%2Cx%2F(x%2B10)%2C0.5*x%2F(0.5*x%2B10)%2C0.3*x%2F(0.3*x%2B10)%2C0.1*x%2F(0.1*x%2B10)&xmin=0&xmax=30&ymin=0&ymax=1&var=x
    Chosen function: 3*x/(3*x+10)
    '''
    PSNR_score_normed = 3 * PSNR_score / (3 * PSNR_score + 10)
    return PSNR_score_normed

def SSIM_norm(SSIM_score):
    '''
    SSIM_score: The SSIM (Structural Similarity Index Measure) score of the image
    Range: -1 to 1, where:
        -1 represents the worst image quality, indicating maximum structural differences.
        1 represents the best image quality, meaning the two images are identical in structure, luminance, and contrast.
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=(x%2B1)%2F2&xmin=-1&xmax=1&ymin=0&ymax=1&var=x
    Chosen function: (x+1)/2
    '''
    SSIM_score_normed = (SSIM_score + 1) / 2
    return SSIM_score_normed

def variance_norm(variance_score):
    '''
    variance_score: reflects the dispersion of the image gray value, that is, the contrast of the image.
    Range: 0 to infinity, where:
        0 represents the simplest image, with minimal variation (i.e., no texture or noise).
        Higher variance values represents the most complex image, with high variation (i.e., a high level of texture or noise).
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=3*x%2F(3*x%2B10)%2Cx%2F(x%2B10)%2C0.5*x%2F(0.5*x%2B10)%2C0.3*x%2F(0.3*x%2B10)%2C0.1*x%2F(0.1*x%2B10)&xmin=0&xmax=30&ymin=0&ymax=1&var=x
    Chosen function: 3*x/(3*x+10)
    '''
    variance_score_normed = 3 * variance_score / (3 * variance_score + 10)
    return variance_score_normed

def improved_aesthetic_predictor_norm(improved_aesthetic_predictor_score):
    '''
    https://github.com/christophschuhmann/improved-aesthetic-predictor

    improved_aesthetic_predictor_score: reflects 
    Range: 0 to infinity, where:
        0 represents 
        Higher variance values represents 
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=5*x%2F(5*x%2B10)%2C3*x%2F(3*x%2B10)%2Cx%2F(x%2B10)%2C0.5*x%2F(0.5*x%2B10)%2C0.3*x%2F(0.3*x%2B10)%2C0.1*x%2F(0.1*x%2B10)&xmin=0&xmax=30&ymin=0&ymax=1&var=x
    Chosen function: 5*x/(5*x+10)
    '''
    improved_aesthetic_predictor_score_normed = 3 * improved_aesthetic_predictor_score / (3 * improved_aesthetic_predictor_score + 10)
    return improved_aesthetic_predictor_score_normed

def skytnt_anime_aesthetic_score_norm(skytnt_anime_aesthetic_score):
    '''
    https://huggingface.co/skytnt/anime-aesthetic/tree/main
    
    skytnt_anime_aesthetic_score: reflects 
    Range: 0 to 1, where:
        0 represents 
        1 represents 
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=x&xmin=0&xmax=1&ymin=0&ymax=1&var=x
    Chosen function: x
    '''
    skytnt_anime_aesthetic_score_normed = skytnt_anime_aesthetic_score
    return skytnt_anime_aesthetic_score_normed

def nsfw_detect_score_norm(nsfw_detect_score):
    '''
    https://huggingface.co/skytnt/anime-aesthetic/tree/main
    
    nsfw_detect_score: reflects 
    Range: 0 to 1, where:
        0 represents safe
        1 represents unsafe
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=1-x&xmin=0&xmax=1&ymin=0&ymax=1&var=x
    Chosen function: 1-x
    '''
    nsfw_detect_score_normed = 1 - nsfw_detect_score
    return nsfw_detect_score_normed

def nsfw_detect_train_score_norm(nsfw_detect_train_score):
    '''
    自训练的三分类黄图检测模型
    
    nsfw_detect_train_score: reflects 
    Range: [0, 1, 2], where:
        0 represents 无风险
        1 represents 中风险
        2 represents 高风险
    
    Function source: https://zh.numberempire.com/graphingcalculator.php?functions=1-0.5*x&xmin=0&xmax=2&ymin=0&ymax=1&var=x
    Chosen function: 1-0.5*x
    '''
    nsfw_detect_train_score_normed = 1 - 0.5 * nsfw_detect_train_score
    return nsfw_detect_train_score_normed