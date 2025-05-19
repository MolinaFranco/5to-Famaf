-- {-# LANGUAGE LambdaCase #-}

module BoolExp
  ( BoolExp (..),
    Relation (..),
    Operator (..),
    interpBoolExp,
  )
where

import IntExp
import State

{-
Gramaticas BoolExp:
⟨boolconst⟩ ::= true | false
⟨boolexp⟩ ::= ⟨boolconst⟩
\| ¬ ⟨boolexp⟩
\| ⟨boolexp⟩ R ⟨boolexp⟩
\| ⟨boolexp⟩ ? ⟨boolexp⟩
R ∈ {<,⩽,=,,,⩾, >}
? ∈ {∧,∨,⇒,⇔}
-}

data BoolExp
  = ConstB Bool
  | NegB BoolExp
  | Rel Relation IntExp IntExp
  | OperationB Operator BoolExp BoolExp
  deriving (Eq, Show)

data Relation = Lt | Leq | Eq | Neq | Gt | Geq
  deriving (Eq, Show)

data Operator = And | Or | Imp | Eqq
  deriving (Eq, Show)

evalRel :: Relation -> Int -> Int -> Bool
evalRel r x y = case r of
  Lt -> x < y
  Leq -> x <= y
  Eq -> x == y
  Neq -> x /= y
  Gt -> x > y
  Geq -> x >= y

evalOp :: Operator -> Bool -> Bool -> Bool
evalOp op b1 b2 = case op of
  And -> b1 && b2
  Or -> b1 || b2
  Imp -> not b1 || b2
  Eqq -> b1 == b2

interpBoolExp :: State -> BoolExp -> Bool
interpBoolExp state expr = case expr of
  ConstB b -> b
  NegB e -> not (interpBoolExp state e)
  Rel r e1 e2 ->
    let v1 = interpIntExp state e1
        v2 = interpIntExp state e2
     in evalRel r v1 v2
  OperationB op e1 e2 ->
    let v1 = interpBoolExp state e1
        v2 = interpBoolExp state e2
     in evalOp op v1 v2
