# Rapport d'Évaluation : Modèle Llama 4 Scout sur le dataset CheXpert

Ce rapport analyse les performances de l'API Groq (Llama 4 Scout) sur un sous-échantillon de 30 images réelles (Chest X-rays) tirées de la base de données Stanford CheXpert.

> [!NOTE]
> Le but pédagogique de cette évaluation est de démontrer les limites d'une IA généraliste lorsqu'elle est utilisée dans un cadre médical pointu sans "fine-tuning".

---

## 1. Évaluation "Baseline" (Prompt basique)

Lors du premier passage, le modèle reçoit une instruction simple d'analyse radiologique.

### Résultats
| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **Précision (Accuracy)** | `33.33%` | Correspond exactement au hasard (1 chance sur 3). |
| **Sensibilité** | `100%` | Le modèle détecte toutes les maladies réelles. |
| **Spécificité** | `0%` | Le modèle n'arrive à identifier **aucun patient sain**. |
| **Taux de classe "Incertain"** | `0%` | Le modèle n'a jamais utilisé la classe "uncertain" (bien que sa confiance interne stagne souvent à 60%). |

### Analyse
Le modèle souffre d'un biais d'**hallucination massive** (Sur-diagnostic / Over-calling). Dès qu'on lui montre une image de poumons (même parfaitement saine), il trouve des taches imaginaires et déclare une opacité suspecte (`suspected_opacity`). Il est incapable de dire qu'un patient n'a rien.

---

## 2. Évaluation "Improved" (Prompt avec Garde-fous)

Pour corriger les hallucinations, le Prompt a été modifié avec des directives strictes :
- *Pénalisation sévère du sur-diagnostic*
- *Obligation de choisir "normal" si les poumons sont clairs*
- *Obligation de choisir "uncertain" si la confiance est sous 60%*

### Résultats
| Métrique | Score | Interprétation |
| :--- | :--- | :--- |
| **Précision (Accuracy)** | `33.33%` | Reste bloquée au niveau du hasard. |
| **Sensibilité** | `0%` | Le modèle ne trouve plus aucune maladie. |
| **Spécificité** | `0%` | Le modèle ne trouve plus aucun patient sain. |
| **Taux d'Incertitude** | `100%` | **Le modèle a répondu `uncertain` à toutes les radios.** |

### Analyse
Le Prompt Engineering a fonctionné *trop bien*. En terrorisant le modèle avec des pénalités sur les faux positifs, Llama 4 Scout a totalement perdu confiance. Sa probabilité interne est passée sous les 60 %, le forçant à utiliser l'étiquette `uncertain` pour les 30 patients. 

> [!IMPORTANT]
> **Le Paradoxe de la Prudence** : Punir une IA non spécialisée l'empêche d'halluciner, mais la paralyse complètement. Elle finit par ne plus prendre aucune décision ("False Negatives").

---

## Conclusion Finale
Ce test scientifique prouve par A+B que :
1. Une IA VLM (Vision-Language Model) classique ne **sait pas** lire une radio.
2. Le Prompt Engineering ne peut pas remplacer le savoir médical ("Fine-Tuning").
3. Il est absolument critique d'inclure des avertissements de sécurité (Guardrails) dans les interfaces médicales, car sans eux, l'IA condamne tous les patients sains par erreur.
