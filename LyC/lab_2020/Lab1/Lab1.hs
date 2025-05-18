{-# LANGUAGE GADTs #-}

data Expr a where
  CInt :: Int       -> Expr Int
  Plus :: Expr Int  -> Expr Int  -> Expr Int
  And  :: Expr Bool -> Expr Bool -> Expr Bool

class DomSem dom where 
   sem :: Expr dom -> dom

instance DomSem Int where
   sem e = undefined

instance DomSem Bool where
   sem e = undefined
