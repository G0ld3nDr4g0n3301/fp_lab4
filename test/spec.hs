{-# LANGUAGE BlockArguments #-}
{-# OPTIONS_GHC -Wno-orphans #-}
{-# OPTIONS_GHC -Wno-unused-imports #-}

import HaskLearn.Metrics
import qualified HaskLearn.Models.KNN as KNN
import qualified HaskLearn.Models.Linreg as Linreg
import qualified HaskLearn.Models.Logreg as Logreg
import qualified HaskLearn.Models.Tree as Tree
import HaskLearn.Preprocessing
import System.Random (mkStdGen)
import Test.Hspec
import Test.QuickCheck

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
              (length trainX + length testX == n)
                && (length trainY + length testY == n)

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
      f1Score (1 :: Int) ([1, 0] :: [Int]) ([1, 0] :: [Int]) `shouldBe` (1.0 :: Double)

    it "F1 Score свойство - всегда в диапазоне [0, 1]" $
      property $ \ys1 ys2 ->
        let len = min (length ys1) (length ys2)
            y1 = take len (ys1 :: [Int])
            y2 = take len (ys2 :: [Int])
            res = f1Score (1 :: Int) y1 y2
         in not (null y1) ==> res >= (0.0 :: Double) && res <= (1.0 :: Double)

  describe "HaskLearn.Models.KNN" $ do
    it "предсказывает ту же точку с k=1" $ do
      let trainX = [[1.0, 2.0], [5.0, 6.0], [10.0, 10.0]] :: [[Double]]
          trainY = [0.0, 1.0, 2.0] :: [Double]
          model = KNN.fit KNN.Classification KNN.Uniform 1 trainX trainY
          predictions = KNN.predict model trainX
      predictions `shouldBe` trainY

  describe "HaskLearn.Models.Linreg" $ do
    it "обучается на прямой y = 2x" $ do
      let trainX = [[1.0], [2.0], [3.0]] :: [[Double]]
          trainY = [2.0, 4.0, 6.0] :: [Double]
          (model, _) = Linreg.fit trainX trainY 0.01 100 (mkStdGen 42)
          preds = Linreg.predict model [[4.0]]
      case preds of
        (p : _) -> p `shouldSatisfy` (\val -> abs (val - 8.0) < 0.5)
        [] -> expectationFailure "Linreg no output"

  describe "HaskLearn.Models.Logreg" $ do
    it "разделяет классы (бинарно)" $ do
      let trainX = [[1.0, 1.0], [10.0, 10.0]] :: [[Double]]
          trainY = [0, 1] :: [Int]
          (model, _) = Logreg.fit trainX trainY 0.1 100 (mkStdGen 42)
          preds = Logreg.predict model [[1.5, 1.5]]
      case preds of
        (p : _) -> p `shouldBe` (0 :: Int)
        [] -> expectationFailure "Logreg no output"

  describe "HaskLearn.Models.Tree" $ do
    it "smoke test, т.е возвращает предсказания" $ do
      let trainX = [[1.0], [2.0], [10.0], [11.0]] :: [[Double]]
          trainY = [1.0, 1.0, 20.0, 20.0] :: [Double]
          model = Tree.fit True trainX trainY 5 2
          preds = Tree.predict model [[1.5]]
      case preds of
        (p : _) -> p `shouldSatisfy` (> (0.0 :: Double))
        [] -> expectationFailure "Tree no output"
