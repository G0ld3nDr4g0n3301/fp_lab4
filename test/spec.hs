import Test.Hspec
import Test.QuickCheck
import Linreg 


main :: IO ()
main = hspec $ do
    
    describe "Unit Tests" $ do
        
        it "test" $ do
            True `shouldBe` True

