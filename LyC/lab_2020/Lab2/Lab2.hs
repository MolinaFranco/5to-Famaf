
{-# LANGUAGE GADTs, TypeSynonymInstances, FlexibleInstances #-}
import Control.Applicative ( liftA, liftA2 )

type Var   = String
type Σ     = Var -> Int

{- Dominios semánticos -}
type MInt  = Maybe Int  -- { n | (n = Just m, m ∈ Int)    ∨ (n = Nothing) }
type MBool = Maybe Bool -- { b | (b = Just b', b' ∈ Bool) ∨ (b = Nothing) }

{- Sintaxis -}
data Expr a where
  CInt :: Int        -> Expr MInt
  Plus :: Expr MInt  -> Expr MInt  -> Expr MInt
  And  :: Expr MBool -> Expr MBool -> Expr MBool

{- Funciones semánticas -}
class DomSem dom where 
   sem :: Expr dom -> Σ -> dom

instance DomSem MInt where
  sem e = undefined

instance DomSem MBool where
  sem e = undefined

{- Funciones auxiliares -}
(-^-) :: (a -> b -> c) -> (Maybe a -> Maybe b -> Maybe c)
(-^-) = liftA2

(-^) :: (a -> b) -> (Maybe a -> Maybe b)
(-^) = liftA
