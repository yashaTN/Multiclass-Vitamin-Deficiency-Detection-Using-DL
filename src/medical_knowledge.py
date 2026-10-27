"""
Clinical Knowledge Base and Early Healthcare Intervention Guidelines
for Multiclass Vitamin Deficiency Detection.
"""

VITAMIN_KNOWLEDGE = {
    "Vitamin A deficiency": {
        "title": "Vitamin A Deficiency (Hypovitaminosis A)",
        "vitamin_name": "Vitamin A (Retinoids / Beta-Carotene)",
        "biological_role": "Essential for rhodopsin synthesis in retinal photoreceptors, epithelial tissue integrity, immune function, and cellular differentiation.",
        "affected_organs": ["Eyes (Cornea & Retina)", "Skin & Mucous Membranes", "Immune System"],
        "clinical_manifestations": [
            "Xerophthalmia (extreme eye dryness) and Night Blindness (nyctalopia)",
            "Bitot's spots (foamy keratinized lesions on conjunctiva)",
            "Keratomalacia (corneal ulceration and liquefaction in advanced stages)",
            "Phrynoderma (follicular hyperkeratosis on arms and legs)",
            "Increased susceptibility to infectious respiratory and enteric diseases"
        ],
        "dietary_sources": [
            "Animal: Beef liver, cod liver oil, pasture-raised egg yolks, fortified whole milk, butter",
            "Plant (Beta-carotene): Sweet potatoes, carrots, spinach, butternut squash, kale, cantaloupe"
        ],
        "recommended_lab_tests": [
            "Serum Retinol (HPLC measurement, normal: 30-80 µg/dL)",
            "Serum Retinol-Binding Protein (RBP)",
            "Dark Adaptation Photometry / Electroretinography"
        ],
        "early_interventions": [
            "Immediate oral Vitamin A supplementation (100,000 to 200,000 IU under medical supervision)",
            "Lubricating artificial tears and antibiotic eye ointment to protect corneal surface",
            "Dietary counseling on pairing provitamin A carotenoids with healthy fats for enhanced absorption"
        ],
        "red_flags": [
            "Clouding, softening, or ulceration of the cornea (high risk of permanent blindness)",
            "Sudden loss of night vision or sharp ocular pain"
        ]
    },

    "Vitamin B complex deficiency": {
        "title": "Vitamin B Complex General Deficiency",
        "vitamin_name": "Vitamin B Complex (B1, B2, B3, B5, B6, B7, B9, B12)",
        "biological_role": "Act as vital coenzymes in cellular respiration, Krebs cycle, fatty acid oxidation, neurotransmitter synthesis, and DNA repair.",
        "affected_organs": ["Nervous System", "Skin & Oral Mucosa", "Gastrointestinal Tract", "Metabolism"],
        "clinical_manifestations": [
            "Chronic generalized fatigue, brain fog, and muscle weakness",
            "Cheilosis and angular stomatitis (cracks and sores at corner of mouth)",
            "Glossitis (swollen, sore, or burning tongue)",
            "Paresthesia (pins and needles in hands and feet)",
            "Seborrheic dermatitis and digestive disturbances"
        ],
        "dietary_sources": [
            "Whole grains (brown rice, oats, quinoa, barley)",
            "Legumes, lentils, chickpeas, and edamame",
            "Eggs, lean poultry, wild salmon, and nutritional yeast",
            "Nuts, seeds, dark leafy greens, and avocados"
        ],
        "recommended_lab_tests": [
            "Comprehensive Metabolic Panel (CMP)",
            "Complete Blood Count (CBC) with differential",
            "Serum B-complex profile or specific B1/B6/B12 assays"
        ],
        "early_interventions": [
            "High-potency B-Complex multivitamin supplementation",
            "Eliminate or substantially reduce alcohol and refined sugar consumption (which deplete B vitamins)",
            "Incorporate diversified whole food meals with activated folate and methylcobalamin"
        ],
        "red_flags": [
            "Unexplained gait unsteadiness, confusion, or severe neuropathic tingling"
        ]
    },

    "Vitamin B-12 deficiency": {
        "title": "Vitamin B-12 Deficiency (Cobalamin Deficiency)",
        "vitamin_name": "Vitamin B-12 (Cobalamin / Methylcobalamin)",
        "biological_role": "Cofactor in methionine synthase and L-methylmalonyl-CoA mutase; crucial for myelin sheath synthesis, neurological function, and erythropoiesis.",
        "affected_organs": ["Peripheral & Central Nervous System", "Bone Marrow (Red Blood Cells)", "Oral Cavity", "Skin"],
        "clinical_manifestations": [
            "Atrophic Glossitis ('Hunter's Glossitis' - beefy, red, smooth depapillated tongue)",
            "Subacute combined degeneration of spinal cord (gait ataxia, vibration/proprioception loss)",
            "Megaloblastic (Macrocytic) anemia presenting with extreme fatigue and pale/jaundice skin",
            "Skin hyperpigmentation over dorsum of fingers and toes",
            "Neuropsychiatric symptoms (memory loss, depression, cognitive decline)"
        ],
        "dietary_sources": [
            "Clams, oysters, wild salmon, tuna, trout",
            "Beef liver, lean grass-fed meats, poultry",
            "Organic eggs, yogurt, cottage cheese, fortified plant milks/nutritional yeast (for vegans)"
        ],
        "recommended_lab_tests": [
            "Serum Vitamin B12 level (<200 pg/mL indicates deficiency, 200-300 pg/mL borderline)",
            "Serum Methylmalonic Acid (MMA) - highly sensitive marker of tissue deficiency",
            "Total Homocysteine level",
            "Intrinsic Factor Antibodies (anti-IF) & Parietal Cell Antibodies to rule out Pernicious Anemia"
        ],
        "early_interventions": [
            "Intramuscular cyanocobalamin / hydroxocobalamin injections (1000 mcg) for malabsorption cases",
            "High-dose sublingual methylcobalamin (1000-2000 mcg daily) for dietary insufficiency",
            "Dietary adjustment for strict vegans and vegetarians with mandatory daily supplementation"
        ],
        "red_flags": [
            "Persistent numbness or shooting electrical sensations in limbs (Lhermitte's sign)",
            "Loss of balance, frequent falls, or rapid cognitive impairment"
        ]
    },

    "Vitamin B2 deficiency": {
        "title": "Vitamin B2 Deficiency (Ariboflavinosis)",
        "vitamin_name": "Vitamin B2 (Riboflavin / FAD / FMN)",
        "biological_role": "Component of flavin adenine dinucleotide (FAD) and flavin mononucleotide (FMN), indispensable for mitochondrial energy production and antioxidant defense.",
        "affected_organs": ["Lips & Oral Mucosa", "Tongue", "Skin", "Eyes (Cornea)"],
        "clinical_manifestations": [
            "Angular Cheilitis (painful erythema and maceration/cracking at oral commissures)",
            "Magenta-colored glossitis with prominent papillae",
            "Nasolabial seborrheic dermatitis and scaly genital eczema",
            "Photophobia, corneal vascularization, and eye burning/fatigue"
        ],
        "dietary_sources": [
            "Dairy products (milk, plain Greek yogurt, artisanal cheeses)",
            "Crimini and portobello mushrooms, almonds, spinach",
            "Eggs, lean meats, salmon, and fortified whole grains"
        ],
        "recommended_lab_tests": [
            "Erythrocyte Glutathione Reductase Activity Coefficient (EGRAC - gold standard)",
            "Urinary Riboflavin Excretion",
            "Complete Blood Count (to evaluate microcytic hypochromic normocytic anemia)"
        ],
        "early_interventions": [
            "Oral Riboflavin replacement (5 to 10 mg daily until symptoms subside)",
            "Topical emollient with antifungal/antibacterial coverage for cracked lip corners",
            "Dietary enhancement with riboflavin-dense dairy and mushroom preparations"
        ],
        "red_flags": [
            "Severe secondary bacterial or fungal (Candida) infection in lip fissures",
            "Corneal neovascularization causing visual blurring"
        ]
    },

    "Vitamin B3 deficieny": {
        "title": "Vitamin B3 Deficiency (Niacin / Pellagra)",
        "vitamin_name": "Vitamin B3 (Niacin / Nicotinic Acid / NAD / NADP)",
        "biological_role": "Constituent of NAD and NADP, carrying hydrogen atoms in over 400 enzymatic oxidation-reduction biochemical pathways.",
        "affected_organs": ["Dermis (Sun-exposed skin)", "Gastrointestinal Tract", "Central Nervous System"],
        "clinical_manifestations": [
            "Pellagra Classic '4 Ds': Dermatitis, Diarrhea, Dementia, and Death (if untreated)",
            "Casal's Necklace (well-demarcated symmetrical hyperpigmented collar rash around neck)",
            "Bilateral sun-exposed erythematous rash on dorsum of hands (Glove distribution)",
            "Bright red, painful, inflamed strawberry tongue and stomatitis",
            "Neurological irritability, tremor, insomnia, and confusion"
        ],
        "dietary_sources": [
            "Organic chicken breast, turkey, yellowfin tuna, wild salmon",
            "Peanuts, peanut butter, sunflower seeds, green peas",
            "Crimini mushrooms, brown rice, whole wheat, fortified grains"
        ],
        "recommended_lab_tests": [
            "Urinary N-methylnicotinamide (NMN) and 2-pyridone excretion",
            "Serum Nicotinamide / NAD/NADH ratio",
            "Liver Function Tests (LFTs) and complete metabolic panel"
        ],
        "early_interventions": [
            "Oral Nicotinamide (100 to 300 mg daily in divided doses; preferred over nicotinic acid to avoid flushing)",
            "Strict sun avoidance and broad-spectrum zinc oxide sunscreen on skin lesions",
            "Correction of concurrent protein (tryptophan) malnutrition"
        ],
        "red_flags": [
            "Acute confusion, hallucinations, or intractable watery diarrhea leading to dehydration"
        ]
    },

    "Vitamin B9 deficiency": {
        "title": "Vitamin B9 Deficiency (Folate Deficiency)",
        "vitamin_name": "Vitamin B9 (Folate / Tetrahydrofolate)",
        "biological_role": "Vital for one-carbon transfers in purine and pyrimidine biosynthesis, amino acid metabolism, and neural tube closure in fetal gestation.",
        "affected_organs": ["Hematopoietic System (Bone Marrow)", "Oral Mucosa", "Fetal Neural System"],
        "clinical_manifestations": [
            "Megaloblastic Anemia (hypersegmented neutrophils and macrocytes)",
            "Sore, red, burning tongue and shallow aphthous ulcers in mouth",
            "Skin pallor, lethargy, breathlessness on minor exertion",
            "Heightened risk of Neural Tube Defects (spina bifida, anencephaly) during pregnancy",
            "Elevated cardiovascular risk secondary to hyperhomocysteinemia"
        ],
        "dietary_sources": [
            "Dark leafy greens (spinach, romaine lettuce, asparagus, Brussels sprouts)",
            "Lentils, black beans, pinto beans, chickpeas",
            "Avocados, oranges, papayas, and fortified breakfast cereals"
        ],
        "recommended_lab_tests": [
            "RBC Folate (reflects tissue folate stores over preceding 3 months; preferred over serum folate)",
            "Serum Folate level (<3 ng/mL indicates deficiency)",
            "Serum Vitamin B12 (MUST be evaluated concurrently to avoid irreversible neurological damage if B12 deficiency is misdiagnosed as folate deficiency)"
        ],
        "early_interventions": [
            "Oral Folic Acid / L-Methylfolate (1 to 5 mg daily under clinical observation)",
            "Periconceptional supplementation (400 to 800 mcg daily) for all women of childbearing age",
            "Avoidance of folate-antagonist drugs without clinical supervision (e.g. methotrexate, trimethoprim)"
        ],
        "red_flags": [
            "Severe symptomatic anemia with chest pain or syncope",
            "Pregnancy with inadequate folate intake"
        ]
    },

    "Vitamin C deficiency": {
        "title": "Vitamin C Deficiency (Scurvy)",
        "vitamin_name": "Vitamin C (L-Ascorbic Acid)",
        "biological_role": "Essential cofactor for prolyl and lysyl hydroxylase in collagen triple-helix stabilization, carnitine biosynthesis, and free radical neutralization.",
        "affected_organs": ["Capillaries & Vascular Endothelium", "Gums & Periodontium", "Skin Hair Follicles", "Musculoskeletal System"],
        "clinical_manifestations": [
            "Scorbutic Gingivitis (spongy, swollen, bleeding purple gums and loose teeth)",
            "Perifollicular hemorrhages and pathognomonic 'Corkscrew' or swan-neck body hairs",
            "Petechiae, purpura, and large subperiosteal hematomas from capillary fragility",
            "Poor wound healing, reopening of old scars, and joint hemarthrosis",
            "Profound lassitude, arthralgia, and microcytic/normocytic anemia"
        ],
        "dietary_sources": [
            "Guavas, kiwi, strawberries, oranges, grapefruits, lemons",
            "Red and yellow bell peppers, broccoli, Brussels sprouts, kale",
            "Tomatoes, acerola cherries, and rose hips"
        ],
        "recommended_lab_tests": [
            "Plasma Ascorbic Acid (<0.2 mg/dL confirms clinical scurvy; normal: 0.4-1.5 mg/dL)",
            "Leukocyte Ascorbic Acid (accurately measures tissue saturation)",
            "Capillary Fragility Test (Rumpel-Leede Tourniquet test)"
        ],
        "early_interventions": [
            "Oral Ascorbic Acid replacement (300 to 1000 mg daily in divided doses; dramatic clinical improvement typically seen within 48 to 72 hours)",
            "Gentle antiseptic oral rinses (chlorhexidine 0.12%) to protect friable gingiva",
            "Incorporate fresh raw or lightly steamed fruits and vegetables daily"
        ],
        "red_flags": [
            "Spontaneous mucosal hemorrhages, hematuria, or excruciating subperiosteal bone pain"
        ]
    },

    "Vitamin D deficiency": {
        "title": "Vitamin D Deficiency (Hypovitaminosis D / Rickets / Osteomalacia)",
        "vitamin_name": "Vitamin D (Cholecalciferol D3 / Ergocalciferol D2)",
        "biological_role": "Maintains calcium and phosphorus homeostasis, promotes intestinal calcium absorption, modulates bone remodeling, and regulates adaptive/innate immunity.",
        "affected_organs": ["Skeletal Bones", "Muscular System", "Parathyroid Gland", "Immune Defense"],
        "clinical_manifestations": [
            "In Children (Rickets): Bowing of legs (genu varum), knock-knees (genu valgum), rachitic rosary (swollen costochondral junctions), and frontal bossing",
            "In Adults (Osteomalacia): Deep, throbbing bone pain (pelvis, lower back, legs) and proximal muscle weakness (difficulty climbing stairs)",
            "Increased risk of low-trauma osteoporotic fractures",
            "Chronic diffuse musculoskeletal aching and frequent respiratory infections"
        ],
        "dietary_sources": [
            "Direct ultraviolet-B (UVB) sunlight exposure (15-20 min midday sunlight on uncovered skin)",
            "Wild-caught oily fish (salmon, sardines, mackerel, herring)",
            "Cod liver oil, fortified whole milk, fortified plant milk, pastured egg yolks",
            "UV-exposed cremini or portobello mushrooms"
        ],
        "recommended_lab_tests": [
            "Serum 25-hydroxyvitamin D [25(OH)D] (Optimal: 30-60 ng/mL, Insufficient: 20-29 ng/mL, Deficient: <20 ng/mL)",
            "Serum Intact Parathyroid Hormone (PTH)",
            "Serum Calcium, Phosphorus, and Alkaline Phosphatase (ALP)",
            "Dual-Energy X-ray Absorptiometry (DEXA) bone density scan"
        ],
        "early_interventions": [
            "Therapeutic loading dose of Vitamin D3 (e.g. 50,000 IU weekly for 8 weeks or 4,000-6,000 IU daily under physician monitoring)",
            "Ensure adequate elemental calcium intake (1,000 to 1,200 mg daily) and magnesium cofactor",
            "Safe sun exposure protocol and follow-up 25(OH)D blood draw at 12 weeks"
        ],
        "red_flags": [
            "Severe bone tenderness, pathological bone fracture, or hypocalcemic tetany/carpopedal spasms"
        ]
    },

    "Vitamin E deficiency": {
        "title": "Vitamin E Deficiency (Tocopherol Deficiency)",
        "vitamin_name": "Vitamin E (Alpha-Tocopherol)",
        "biological_role": "Primary lipid-soluble chain-breaking antioxidant protecting polyunsaturated fatty acids in cell membranes from lipid peroxidation.",
        "affected_organs": ["Central & Peripheral Nervous System", "Retina", "Red Blood Cells"],
        "clinical_manifestations": [
            "Progressive spinocerebellar ataxia and loss of deep tendon reflexes (hyporeflexia)",
            "Loss of vibratory sensation and proprioception (posterior column dysfunction)",
            "Pigmentary retinopathy and ophthalmoplegia (restricted eye movement)",
            "Hemolytic anemia due to red blood cell membrane fragility"
        ],
        "dietary_sources": [
            "Wheat germ oil, sunflower seeds, almonds, hazelnuts",
            "Avocados, peanut butter, olive oil, spinach, broccoli"
        ],
        "recommended_lab_tests": [
            "Serum Alpha-Tocopherol (normal: 5-20 µg/mL)",
            "Serum Tocopherol-to-Lipid ratio (especially informative if hyperlipidemia is present)",
            "Neurological electromyography (EMG) and nerve conduction studies"
        ],
        "early_interventions": [
            "Oral Alpha-tocopherol supplementation (100 to 400 IU daily; higher doses for genetic abetalipoproteinemia)",
            "Clinical evaluation for underlying fat malabsorption (cystic fibrosis, celiac disease, cholestasis)",
            "Dietary intake of cold-pressed oils, raw seeds, and nuts"
        ],
        "red_flags": [
            "Progressive ataxia, inability to walk steadily, or rapid visual field constriction"
        ]
    },

    "Vitamin K deficiency": {
        "title": "Vitamin K Deficiency (Coagulopathy)",
        "vitamin_name": "Vitamin K (Phylloquinone K1 / Menaquinone K2)",
        "biological_role": "Essential cofactor for gamma-glutamyl carboxylase, required for post-translational activation of clotting factors II (prothrombin), VII, IX, X, and proteins C and S.",
        "affected_organs": ["Circulatory System (Coagulation Cascade)", "Bone Matrix (Osteocalcin)"],
        "clinical_manifestations": [
            "Easy ecchymosis (widespread, unexplained bruising) and hematomas",
            "Mucosal bleeding: frequent epistaxis (nosebleeds), gingival oozing, heavy menorrhagia",
            "Prolonged bleeding from minor cuts, punctures, or surgical sites",
            "Gastrointestinal bleeding (melena) or hematuria",
            "Vitamin K Deficiency Bleeding (VKDB) in infants"
        ],
        "dietary_sources": [
            "Phylloquinone (K1): Collard greens, kale, spinach, turnip greens, broccoli, Brussels sprouts, parsley",
            "Menaquinone (K2): Natto (fermented soy), grass-fed aged gouda/cheddar, organ meats, pasture-raised egg yolks"
        ],
        "recommended_lab_tests": [
            "Prothrombin Time (PT) and International Normalized Ratio (INR) - prolonged in deficiency",
            "Activated Partial Thromboplastin Time (aPTT)",
            "Under-carboxylated Prothrombin (PIVKA-II - Protein Induced by Vitamin K Absence)",
            "Specific factor assays (Factors II, VII, IX, X)"
        ],
        "early_interventions": [
            "Oral or subcutaneous Vitamin K1 (Phytonadione 1 to 10 mg) guided by INR",
            "Assessment of broad-spectrum antibiotic use (which eradicates K2-producing gut flora) or anticoagulant therapy",
            "Increase intake of dark green leafy vegetables while maintaining stable intake if on prescribed warfarin"
        ],
        "red_flags": [
            "Active gastrointestinal hemorrhage, blood in urine or stool, or intracranial bleeding symptoms"
        ]
    },

    "zinc, iron, biotin, or protein deficiency": {
        "title": "Trace Mineral / Protein Deficiency (Zinc, Iron, Biotin, Protein)",
        "vitamin_name": "Zinc (Zn), Iron (Fe), Biotin (Vitamin B7), Amino Acids (Protein)",
        "biological_role": "Synergistic micronutrients and building blocks vital for keratinization, hemoglobin oxygen transport, zinc-finger transcription factors, and carboxylase enzymes.",
        "affected_organs": ["Nails & Hair Follicles", "Dermis", "Erythrocytes", "Immune Defense"],
        "clinical_manifestations": [
            "Koilonychia ('Spoon nails' - concave, thin, brittle nails typical of iron deficiency)",
            "Beau's lines, leukonychia (white transverse marks), and slow nail plate growth",
            "Diffuse Telogen Effluvium (hair shedding, premature thinning, alopecia)",
            "Acrodermatitis enteropathica (perioral and acral erythematous crusting plaques from zinc deficiency)",
            "Fatigue, pallor, cold intolerance (microcytic hypochromic iron-deficiency anemia)",
            "Edema, muscle wasting, and flag sign in hair from protein energy malnutrition"
        ],
        "dietary_sources": [
            "Zinc: Wild oysters, pumpkin seeds, grass-fed beef, cashews, lentils",
            "Iron: Grass-fed red meat, organ meats, spinach, black beans, blackstrap molasses",
            "Biotin: Cooked egg yolks, salmon, almonds, sunflower seeds, sweet potatoes",
            "Protein: Wild fish, pasture-raised poultry, eggs, tempeh, Greek yogurt, legumes"
        ],
        "recommended_lab_tests": [
            "Serum Ferritin (most sensitive test for iron deficiency stores; <30 ng/mL diagnostic)",
            "Total Iron Binding Capacity (TIBC) and Transferrin Saturation",
            "Serum Zinc level (fasting morning sample)",
            "Serum Albumin and Total Protein (assesses systemic protein status)",
            "Urinary 3-hydroxyisovalerate (biotin functional marker)"
        ],
        "early_interventions": [
            "Targeted oral micronutrient replacement: Ferrous bisglycinate with Vitamin C for optimal absorption; Zinc gluconate/picolinate (15-30 mg daily)",
            "Oral Biotin supplementation (2.5 to 5 mg daily for onychorrhexis and hair brittleness)",
            "Optimizing daily high-biological-value protein intake (1.2 to 1.6 g/kg body weight)",
            "Rule out gastrointestinal celiac disease or occult bleeding"
        ],
        "red_flags": [
            "Hemoglobin < 7-8 g/dL requiring urgent transfusion evaluation",
            "Rapid onset ulcerating acral skin lesions or extreme hypoproteinemic ascites"
        ]
    }
}
