module HaskLearn.Metrics where

meanAbsoluteError :: [Double] -> [Double] -> Double
meanAbsoluteError yTrue yPred = 
    sum (zipWith (\yt yp -> abs (yt - yp)) yTrue yPred) / fromIntegral (length yTrue)

meanSquaredError :: [Double] -> [Double] -> Double
meanSquaredError yTrue yPred = 
    sum (zipWith (\yt yp -> (yt - yp) ^ 2) yTrue yPred) / fromIntegral (length yTrue)
