{-# OPTIONS_GHC -Wno-orphans #-}
import Test.Hspec
import Test.QuickCheck
import System.Random (mkStdGen)
import HaskLearn.Metrics
import HaskLearn.Preprocessing

main :: IO ()
main = hspec spec

spec :: Spec
spec = do
  describe "HaskLearn.Preprocessing.trainTestSplit" $ do
    it "сумма размеров выборок должна быть равна исходному размеру" $
      property $ \xs ys -> 
        let n = min (length (xs :: [Int])) (length (ys :: [Int]))
            xs' = take n xs
            ys' = take n ys
            testRatio = 0.2 :: Double
            gen = mkStdGen 42
            (trainX, testX, trainY, testY) = trainTestSplit gen xs' ys' testRatio
        in n > 0 ==> 
           (length trainX + length testX == n) && 
           (length trainY + length testY == n)

    it "процент тестовой выборки должен соответствовать переданному testSize" $ do
      let xs = replicate 100 [1.0 :: Double]
          ys = replicate 100 (1.0 :: Double)
          gen = mkStdGen 42
          testRatio = 0.25 :: Double
          (_, testX, _, _) = trainTestSplit gen xs ys testRatio
      length testX `shouldBe` 25

  describe "HaskLearn.Metrics (регрессия)" $ do
    let yTrue = [1.0, 2.0, 3.0] :: [Double]
    let yPred = [1.0, 2.0, 3.0] :: [Double]
    let yNoise = [1.1, 1.9, 3.2] :: [Double]

    it "MSE идеального предсказания должен быть 0" $
      mse yTrue yPred `shouldBe` (0.0 :: Double)

    it "MSE зашумленного предсказания должен быть положителен" $
      mse yTrue yNoise `shouldSatisfy` (> (0.0 :: Double))

    it "MAE свойство - mae [a] [b] == mae [b] [a]" $
      property $ \xs ys -> 
        let len = min (length xs) (length ys)
            y1 = take len (map abs xs) :: [Double]
            y2 = take len (map abs ys) :: [Double]
        in not (null y1) ==> mae y1 y2 `shouldBe` mae y2 y1

  describe "HaskLearn.Metrics (классификация)" $ do
    let labelsTrue = [1, 1, 0, 0] :: [Int]
    let labelsPred = [1, 0, 0, 1] :: [Int] 

    it "правильно считает Accuracy" $
      accuracyScore labelsTrue labelsPred `shouldBe` (0.5 :: Double)

    it "матрица ошибок должна давать (1,1,1,1) для данного примера" $
      getConfusionMatrix (1 :: Int) labelsTrue labelsPred `shouldBe` (1, 1, 1, 1)

    it "Precision для posLabel 1" $
      precisionScore (1 :: Int) labelsTrue labelsPred `shouldBe` (0.5 :: Double)

    it "F1 Score должен быть корректным для идеального совпадения" $
      f1Score (1 :: Int) ([1,0] :: [Int]) ([1,0] :: [Int]) `shouldBe` (1.0 :: Double)

    it "F1 Score свойство - всегда в диапазоне [0, 1]" $
      property $ \ys1 ys2 ->
        let len = min (length ys1) (length ys2)
            y1 = take len (ys1 :: [Int])
            y2 = take len (ys2 :: [Int])
            res = f1Score (1 :: Int) y1 y2
        in not (null y1) ==> res >= (0.0 :: Double) && res <= (1.0 :: Double)