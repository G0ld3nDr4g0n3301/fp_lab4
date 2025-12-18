module HaskLearn.Metrics where

meanAbsoluteError :: [Double] -> [Double] -> Double
meanAbsoluteError yTrue yPred = 
    sum (zipWith (\yt yp -> abs (yt - yp)) yTrue yPred) / fromIntegral (length yTrue)

meanSquaredError :: [Double] -> [Double] -> Double
meanSquaredError yTrue yPred = 
    sum (zipWith (\yt yp -> (yt - yp) ^ (2 :: Int)) yTrue yPred) / fromIntegral (length yTrue)

rmse :: [Double] -> [Double] -> Double
rmse yTrue yPred = sqrt (meanSquaredError yTrue yPred)

r2Score :: [Double] -> [Double] -> Double
r2Score yTrue yPred = 
    let yMean = sum yTrue / fromIntegral (length yTrue)
        ssTotal = sum $ map (\y -> (y - yMean) ^ (2 :: Int)) yTrue
        ssRes   = sum $ zipWith (\yt yp -> (yt - yp) ^ (2 :: Int)) yTrue yPred
    in 1 - (ssRes / ssTotal)

