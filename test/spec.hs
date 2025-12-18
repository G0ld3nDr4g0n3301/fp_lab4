import Test.Hspec


main :: IO ()
main = hspec $ do
    
    describe "Unit Tests" $ do
        
        it "test" $ do
            True `shouldBe` True

