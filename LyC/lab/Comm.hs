-- {-# LANGUAGE LambdaCase #-}

module Comm
  ( Comm (..),
    interpComm,
  )
where

import BoolExp
import IntExp
import State

{-
Gramaticas Comandos:
⟨comm⟩ ::= skip
\| ⟨var⟩ := ⟨intexp⟩
\| ⟨comm⟩ ; ⟨comm⟩
\| if ⟨boolexp⟩ then ⟨comm⟩ else ⟨comm⟩
\| newvar ⟨var⟩ := ⟨intexp⟩ in ⟨comm⟩
\| while ⟨boolexp⟩ do ⟨comm⟩
\| fail
-}

type Var = String

data Comm
  = Skip
  | Assign Var IntExp
  | Seq Comm Comm
  | Condicional BoolExp Comm Comm
  | NewVar Var IntExp Comm
  | While BoolExp Comm
  deriving (Eq, Show)

interpComm :: State -> Comm -> State
interpComm state comm = case comm of
  Skip -> state
  Assign v ie -> interpComm (updateState v (interpIntExp state ie) state) comm
  Seq c1 c2 -> interpComm (interpComm state c1) c2
  Condicional b ct cf -> if interpBoolExp state b then interpComm state ct else interpComm state cf
  NewVar v ie c -> updateState v (state v) (interpComm (updateState v (interpIntExp state ie) state) c)
  While be body ->
    if interpBoolExp state be
      then interpComm (interpComm state body) (While be body)
      else state

  -- el bucle se ejecuta mientras la condición sea verdadera, 
  -- y cada vez lo hace con el nuevo estado que generó la iteración anterior.