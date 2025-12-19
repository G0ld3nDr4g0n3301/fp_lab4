{-# OPTIONS_GHC -Wno-unused-top-binds #-}

module HaskLearn.Models.KNN 
    ( KNNMode(..), KNNWeights(..), KNNModel
    , fit, predict
    ) where

import qualified Data.Vector.Unboxed as VU
import qualified Data.Vector as V
import Data.List (sortOn, maximumBy, group, sort)
import Data.Ord (comparing)
import Data.Maybe ()

type VectorD = VU.Vector Double

data KNNMode = Classification | Regression deriving (Eq, Show)
data KNNWeights = Uniform | Distance deriving (Eq, Show)

data KDTree = KDNode 
    { _nodePoint  :: !VectorD
    , _nodeTarget :: !Double
    , _nodeAxis   :: !Int
    , _leftChild  :: !KDTree
    , _rightChild :: !KDTree
    } | KDEmpty

data KNNModel = KNNModel
    { _knnTree    :: !KDTree
    , _knnk       :: !Int
    , _knnWeights :: !KNNWeights
    , _knnMode    :: !KNNMode
    }



fit :: KNNMode -> KNNWeights -> Int -> [[Double]] -> [Double] -> KNNModel
fit m w k kf kt = 
    let samples = V.fromList $ zipWith (\f t -> (VU.fromList f, t)) kf kt
    in KNNModel (buildTree 0 samples) k w m

buildTree :: Int -> V.Vector (VectorD, Double) -> KDTree
buildTree depth samples
    | V.null samples = KDEmpty
    | otherwise =
        let dims = VU.length (fst $ samples V.! 0)
            axis = depth `mod` dims            
            listSorted = sortOn (\(v, _) -> v VU.! axis) (V.toList samples)
            medianIdx = length listSorted `div` 2
            
            splitResult = splitAt medianIdx listSorted
        in case splitResult of
            (leftL, (pivotV, pivotT) : rightL) ->
                KDNode 
                    { _nodePoint  = pivotV
                    , _nodeTarget = pivotT
                    , _nodeAxis   = axis
                    , _leftChild  = buildTree (depth + 1) (V.fromList leftL)
                    , _rightChild = buildTree (depth + 1) (V.fromList rightL)
                    }
            (leftL, []) -> 
                case reverse leftL of
                    ((pV, pT):rest) -> KDNode pV pT axis (buildTree (depth+1) (V.fromList (reverse rest))) KDEmpty
                    [] -> KDEmpty


distSq :: VectorD -> VectorD -> Double
distSq v1 v2 = VU.sum $ VU.zipWith (\a b -> let d = a - b in d * d) v1 v2

findKNearest :: Int -> KDTree -> VectorD -> [(Double, Double)]
findKNearest k root target = take k $ search root []
  where
    search KDEmpty currentBest = currentBest
    search node currentBest =
        let axis = _nodeAxis node
            point = _nodePoint node
            d2 = distSq point target
            
            updatedBest = insertSorted (d2, _nodeTarget node) currentBest
            
            (near, far) = if (target VU.! axis) < (point VU.! axis)
                          then (_leftChild node, _rightChild node)
                          else (_rightChild node, _leftChild node)
            
            bestAfterNear = search near updatedBest
            
            distToPlaneSq = let d = (target VU.! axis) - (point VU.! axis) in d * d
            
        in if length bestAfterNear < k || distToPlaneSq < fst (last bestAfterNear)
           then search far bestAfterNear
           else bestAfterNear

    insertSorted new [] = [new]
    insertSorted new@(d, _) list = 
        let (lt, gt) = span ((< d) . fst) list
        in take k (lt ++ [new] ++ gt)


predict :: KNNModel -> [[Double]] -> [Double]
predict (KNNModel tree k w mode) =
    map (processOne . VU.fromList)
  where
    processOne query =
        let nearest = findKNearest k tree query
        in case mode of
            Regression     -> aggregateReg w nearest
            Classification -> aggregateClas w nearest

aggregateReg :: KNNWeights -> [(Double, Double)] -> Double
aggregateReg _ [] = 0.0
aggregateReg Uniform ns = sum (map snd ns) / fromIntegral (length ns)
aggregateReg Distance ns =
    let eps = 1e-10
        weightsList = map (\(d2, y) -> (y / (sqrt d2 + eps), 1 / (sqrt d2 + eps))) ns
        sumWeightedY = sum $ map fst weightsList
        sumW = sum $ map snd weightsList
    in sumWeightedY / sumW

aggregateClas :: KNNWeights -> [(Double, Double)] -> Double
aggregateClas _ [] = 0.0
aggregateClas Uniform ns = 
    let labels = map (round . snd) ns :: [Int]
        grouped = group (sort labels)
        best = maximumBy (comparing length) grouped
    in case best of
        (x:_) -> fromIntegral x
        []    -> 0.0
aggregateClas Distance ns =
    let eps = 1e-10
        scores = foldl' (\acc (d2, y) -> 
            let w = 1 / (sqrt d2 + eps)
            in updateScore y w acc) [] ns
    in case scores of
        [] -> 0.0
        _  -> fst $ maximumBy (comparing snd) scores
  where
    updateScore y w [] = [(y, w)]
    updateScore y w ((val, score):xs)
        | y == val = (val, score + w) : xs
        | otherwise = (val, score) : updateScore y w xs