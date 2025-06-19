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
  | Fail
  deriving (Eq, Show)

interpComm :: Comm -> State -> Maybe State
interpComm comm state = case comm of
  Skip -> Just state
  Fail -> Nothing
  Assign v ie -> Just (updateState v (interpIntExp ie state) state)
  Seq c1 c2 ->
    case interpComm c1 state of
      Nothing -> Nothing
      Just s' -> interpComm c2 s'
  Condicional b ct cf ->
    if interpBoolExp b state
      then interpComm ct state
      else interpComm cf state
  NewVar v ie c ->
    let initial = interpIntExp ie state
        oldVal = state v
        updatedState = updateState v initial state
     in case interpComm c updatedState of
          Nothing -> Nothing
          Just s' -> Just (updateState v oldVal s')
  While be body ->
    let evalWhile s =
          if interpBoolExp be s
            then case interpComm body s of
              Nothing -> Nothing
              Just s' -> evalWhile s'
            else Just s
     in evalWhile state


