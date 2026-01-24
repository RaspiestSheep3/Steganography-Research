from Helpers.HelperFunctions import SplitIntoBlocks
from Helpers.SteganalysisMethods import ChiSquareAttack, ZhangLSBMatching, SamplePairAnalysis, PSNR

coverPath = input("Cover : ").strip('"')
stegoPath = input("Stego : ").strip('"')

coverBlocks = SplitIntoBlocks(coverPath, 256)[0][0]
stegoBlocks = SplitIntoBlocks(stegoPath, 256)[0][0]
print(f"Cover Chi2  : {ChiSquareAttack(coverBlocks)} | Stego Chi2 : {ChiSquareAttack(stegoBlocks)}")
print(f"Cover SPA   : {SamplePairAnalysis(coverBlocks)} | Stego SPA  : {SamplePairAnalysis(stegoBlocks)}")
print(f"Cover Zhang : {ZhangLSBMatching(coverBlocks)} | Stego Zhang : {ZhangLSBMatching(stegoBlocks)}")
print(f"PSNR : {PSNR(coverBlocks, stegoBlocks)}")