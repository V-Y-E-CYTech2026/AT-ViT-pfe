import json

# Votre chaîne JSON (j'ai tronqué la liste pour la lisibilité du code, 
# vous pouvez coller l'intégralité de votre JSON ici)
json_data = """
{
    "samples": [
        {
            "code": "castanea180",
            "iou": 0.2850663320186447
        },
        {
            "code": "ulmus9",
            "iou": 0.4216442583204761
        },
        {
            "code": "ulmus545",
            "iou": 0.19242082468645674
        },
        {
            "code": "monimiaceae97",
            "iou": 0.7343133462282398
        },
        {
            "code": "litsea74",
            "iou": 0.8808811247216035
        },
        {
            "code": "magnolia34",
            "iou": 0.3053907977397957
        },
        {
            "code": "magnolia315",
            "iou": 0.810516548097742
        },
        {
            "code": "magnolia70",
            "iou": 0.2697005861911615
        },
        {
            "code": "ulmus178",
            "iou": 0.21866545651301206
        },
        {
            "code": "eugenia170",
            "iou": 0.778721876495931
        },
        {
            "code": "castanea27",
            "iou": 0.10811925207507933
        },
        {
            "code": "magnolia233",
            "iou": 0.8145735802971708
        },
        {
            "code": "litsea148",
            "iou": 0.557960950902704
        },
        {
            "code": "magnolia579",
            "iou": 0.4502251125562781
        },
        {
            "code": "magnolia250",
            "iou": 0.07958368926622618
        },
        {
            "code": "castanea53",
            "iou": 0.6950327300731614
        },
        {
            "code": "eugenia533",
            "iou": 0.28819194033526235
        },
        {
            "code": "laurus400",
            "iou": 0.6992308951071837
        },
        {
            "code": "ulmus92",
            "iou": 0.47704396389375936
        },
        {
            "code": "eugenia536",
            "iou": 0.27364823629927987
        },
        {
            "code": "monimiaceae102",
            "iou": 0.13492784513784548
        },
        {
            "code": "magnolia128",
            "iou": 0.8098755011605824
        },
        {
            "code": "monimiaceae220",
            "iou": 0.8091882750845547
        },
        {
            "code": "monimiaceae154",
            "iou": 0.21264253573558378
        },
        {
            "code": "rubus534",
            "iou": 0.6755348495065285
        },
        {
            "code": "laurus19",
            "iou": 0.7805905153849835
        },
        {
            "code": "laurus177",
            "iou": 0.6085818181818182
        },
        {
            "code": "eugenia560",
            "iou": 0.74364255723679
        },
        {
            "code": "ulmus201",
            "iou": 0.5901921896397034
        },
        {
            "code": "rubus270",
            "iou": 0.27512821407389315
        },
        {
            "code": "ulmus5",
            "iou": 0.25406000069849477
        },
        {
            "code": "laurus286",
            "iou": 0.2567428201856351
        },
        {
            "code": "ulmus222",
            "iou": 0.20864355412294378
        },
        {
            "code": "rubus61",
            "iou": 0.10606647419195832
        },
        {
            "code": "convolvulaceae15",
            "iou": 0.05121993094370577
        },
        {
            "code": "laurus476",
            "iou": 0.2867344525892646
        },
        {
            "code": "eugenia247",
            "iou": 0.2589952002242231
        },
        {
            "code": "magnolia123",
            "iou": 0.2725149965779621
        },
        {
            "code": "desmodium14",
            "iou": 0.0713303665110894
        },
        {
            "code": "monimiaceae679",
            "iou": 0.17829876241938294
        },
        {
            "code": "desmodium21",
            "iou": 0.13637314369378792
        },
        {
            "code": "ulmus85",
            "iou": 0.3849318955732123
        },
        {
            "code": "eugenia220",
            "iou": 0.07301603850631636
        },
        {
            "code": "ulmus159",
            "iou": 0.21077442248817144
        },
        {
            "code": "monimiaceae134",
            "iou": 0.6909752868490733
        },
        {
            "code": "eugenia44",
            "iou": 0.297974773494404
        },
        {
            "code": "monimiaceae528",
            "iou": 0.704299572837239
        },
        {
            "code": "ulmus13",
            "iou": 0.3765019443352121
        },
        {
            "code": "rubus10",
            "iou": 0.22404780987559675
        },
        {
            "code": "castanea82",
            "iou": 0.15714885576317447
        },
        {
            "code": "desmodium452",
            "iou": 0.6538099717779868
        },
        {
            "code": "eugenia551",
            "iou": 0.4748256376317263
        },
        {
            "code": "laurus107",
            "iou": 0.18371436537255176
        },
        {
            "code": "laurus384",
            "iou": 0.30126991088694405
        },
        {
            "code": "convolvulaceae468",
            "iou": 0.3006677651548001
        },
        {
            "code": "monimiaceae145",
            "iou": 0.25936718651628804
        },
        {
            "code": "laurus460",
            "iou": 0.3003807668467192
        },
        {
            "code": "desmodium84",
            "iou": 0.26134721017635154
        },
        {
            "code": "laurus437",
            "iou": 0.2059071581684647
        },
        {
            "code": "litsea147",
            "iou": 0.21693482172492706
        },
        {
            "code": "castanea334",
            "iou": 0.8037959554429411
        },
        {
            "code": "convolvulaceae165",
            "iou": 0.3262369496141625
        },
        {
            "code": "desmodium496",
            "iou": 0.49253908804470603
        },
        {
            "code": "laurus65",
            "iou": 0.0988274413720686
        },
        {
            "code": "ulmus271",
            "iou": 0.06752732623318386
        },
        {
            "code": "ulmus314",
            "iou": 0.7195797361133736
        },
        {
            "code": "monimiaceae265",
            "iou": 0.7718538352496456
        },
        {
            "code": "castanea336",
            "iou": 0.30814787900808427
        },
        {
            "code": "laurus292",
            "iou": 0.1591335613435061
        },
        {
            "code": "ulmus282",
            "iou": 0.8266656288916563
        },
        {
            "code": "desmodium62",
            "iou": 0.21930105803142033
        },
        {
            "code": "monimiaceae185",
            "iou": 0.3813376107120441
        },
        {
            "code": "rubus111",
            "iou": 0.4105586878523834
        },
        {
            "code": "desmodium35",
            "iou": 0.16594872872452196
        },
        {
            "code": "ulmus70",
            "iou": 0.46446673891005696
        },
        {
            "code": "litsea28",
            "iou": 0.22489417274610465
        },
        {
            "code": "castanea140",
            "iou": 0.3057764937257427
        },
        {
            "code": "litsea215",
            "iou": 0.20144280736076298
        },
        {
            "code": "litsea145",
            "iou": 0.09103277674706246
        },
        {
            "code": "magnolia17",
            "iou": 0.27610285774138926
        },
        {
            "code": "castanea330",
            "iou": 0.22452173913043477
        },
        {
            "code": "rubus60",
            "iou": 0.356099776411403
        },
        {
            "code": "monimiaceae167",
            "iou": 0.2820818573983797
        },
        {
            "code": "monimiaceae537",
            "iou": 0.2231152800055792
        },
        {
            "code": "desmodium75",
            "iou": 0.40755398778284435
        },
        {
            "code": "rubus537",
            "iou": 0.5293331104796924
        },
        {
            "code": "eugenia191",
            "iou": 0.3814217075274116
        },
        {
            "code": "ulmus88",
            "iou": 0.20303731762065094
        },
        {
            "code": "rubus533",
            "iou": 0.4090660211521282
        },
        {
            "code": "convolvulaceae179",
            "iou": 0.12542372881355932
        },
        {
            "code": "ulmus155",
            "iou": 0.25538445348674055
        },
        {
            "code": "eugenia561",
            "iou": 0.50653340256144
        },
        {
            "code": "rubus92",
            "iou": 0.2642846862406448
        },
        {
            "code": "ulmus274",
            "iou": 0.6642135889098066
        },
        {
            "code": "ulmus199",
            "iou": 0.580954158114076
        },
        {
            "code": "ulmus131",
            "iou": 0.5445761967501098
        },
        {
            "code": "eugenia67",
            "iou": 0.23882907606783477
        },
        {
            "code": "laurus17",
            "iou": 0.1858685947374871
        },
        {
            "code": "desmodium86",
            "iou": 0.05413630593186442
        },
        {
            "code": "amborella38",
            "iou": 0.23437091745196748
        },
        {
            "code": "eugenia694",
            "iou": 0.6252293577981651
        },
        {
            "code": "ulmus202",
            "iou": 0.20035149384885764
        },
        {
            "code": "monimiaceae495",
            "iou": 0.14064655849939744
        },
        {
            "code": "amborella77",
            "iou": 0.24634120475113122
        },
        {
            "code": "magnolia112",
            "iou": 0.5351647198713017
        },
        {
            "code": "eugenia115",
            "iou": 0.11389048790273344
        },
        {
            "code": "litsea21",
            "iou": 0.11916157205240174
        },
        {
            "code": "amborella73",
            "iou": 0.15501862731050295
        },
        {
            "code": "laurus474",
            "iou": 0.4467933660758773
        },
        {
            "code": "litsea0",
            "iou": 0.7652030217186024
        },
        {
            "code": "monimiaceae353",
            "iou": 0.3240961644338411
        },
        {
            "code": "desmodium23",
            "iou": 0.5186477382098171
        },
        {
            "code": "castanea68",
            "iou": 0.03870238302313446
        },
        {
            "code": "litsea113",
            "iou": 0.06150194593273879
        },
        {
            "code": "rubus81",
            "iou": 0.15207252984373368
        },
        {
            "code": "convolvulaceae118",
            "iou": 0.07248518044719915
        },
        {
            "code": "eugenia33",
            "iou": 0.32511421895162695
        },
        {
            "code": "rubus588",
            "iou": 0.763187051399393
        },
        {
            "code": "rubus77",
            "iou": 0.6922493361992667
        },
        {
            "code": "litsea424",
            "iou": 0.09676801115079711
        },
        {
            "code": "laurus435",
            "iou": 0.1578928903467995
        },
        {
            "code": "litsea200",
            "iou": 0.6052925235134704
        },
        {
            "code": "convolvulaceae160",
            "iou": 0.06321678809216719
        },
        {
            "code": "laurus359",
            "iou": 0.7369418132611637
        },
        {
            "code": "ulmus177",
            "iou": 0.4959183673469388
        },
        {
            "code": "castanea331",
            "iou": 0.5984126984126984
        },
        {
            "code": "rubus282",
            "iou": 0.7485447450905136
        },
        {
            "code": "monimiaceae579",
            "iou": 0.765160946114174
        },
        {
            "code": "monimiaceae171",
            "iou": 0.7322958559188669
        },
        {
            "code": "ulmus322",
            "iou": 0.5547102622207835
        },
        {
            "code": "monimiaceae47",
            "iou": 0.3092122458225157
        },
        {
            "code": "monimiaceae86",
            "iou": 0.07887674397938328
        },
        {
            "code": "desmodium37",
            "iou": 0.42072458122321776
        },
        {
            "code": "monimiaceae76",
            "iou": 0.659330985915493
        },
        {
            "code": "magnolia73",
            "iou": 0.7607954545454545
        },
        {
            "code": "amborella7",
            "iou": 0.30581547879317883
        },
        {
            "code": "desmodium481",
            "iou": 0.20774920443794617
        },
        {
            "code": "eugenia149",
            "iou": 0.2357730757003073
        },
        {
            "code": "litsea70",
            "iou": 0.31772335553341374
        },
        {
            "code": "ulmus116",
            "iou": 0.31853795094811344
        },
        {
            "code": "eugenia32",
            "iou": 0.28502076797700626
        },
        {
            "code": "convolvulaceae82",
            "iou": 0.4087271889931495
        },
        {
            "code": "rubus34",
            "iou": 0.37908004066893386
        },
        {
            "code": "monimiaceae65",
            "iou": 0.16060733424398446
        },
        {
            "code": "ulmus165",
            "iou": 0.29892326991427975
        },
        {
            "code": "castanea332",
            "iou": 0.7069759878584205
        },
        {
            "code": "laurus62",
            "iou": 0.40871721257992955
        },
        {
            "code": "laurus431",
            "iou": 0.485160666468313
        },
        {
            "code": "litsea116",
            "iou": 0.4671977260365308
        },
        {
            "code": "desmodium42",
            "iou": 0.6354406389603358
        },
        {
            "code": "magnolia93",
            "iou": 0.3087260610055293
        },
        {
            "code": "ulmus6",
            "iou": 0.25010945900979004
        },
        {
            "code": "convolvulaceae197",
            "iou": 0.20352470930232558
        },
        {
            "code": "monimiaceae250",
            "iou": 0.1958151974408726
        },
        {
            "code": "monimiaceae89",
            "iou": 0.15350332943006448
        },
        {
            "code": "convolvulaceae466",
            "iou": 0.0594186902133922
        },
        {
            "code": "castanea86",
            "iou": 0.7935147044736086
        },
        {
            "code": "rubus22",
            "iou": 0.6492890995260664
        },
        {
            "code": "laurus63",
            "iou": 0.25828460038986356
        },
        {
            "code": "ulmus125",
            "iou": 0.6579605445957211
        },
        {
            "code": "desmodium94",
            "iou": 0.30846895117757606
        },
        {
            "code": "ulmus50",
            "iou": 0.2271310947482027
        },
        {
            "code": "monimiaceae4",
            "iou": 0.3701501932054622
        },
        {
            "code": "monimiaceae582",
            "iou": 0.6978917341552182
        },
        {
            "code": "monimiaceae531",
            "iou": 0.5752794901264235
        },
        {
            "code": "laurus172",
            "iou": 0.2626432551782147
        },
        {
            "code": "ulmus234",
            "iou": 0.7649698593436035
        },
        {
            "code": "monimiaceae547",
            "iou": 0.8936385579693218
        },
        {
            "code": "magnolia593",
            "iou": 0.5473079968329374
        },
        {
            "code": "eugenia25",
            "iou": 0.2316447174653932
        },
        {
            "code": "ulmus117",
            "iou": 0.28003056457635067
        },
        {
            "code": "magnolia69",
            "iou": 0.3555621904527732
        },
        {
            "code": "desmodium286",
            "iou": 0.08380053626771598
        },
        {
            "code": "monimiaceae700",
            "iou": 0.39257164799106364
        },
        {
            "code": "rubus57",
            "iou": 0.264330985915493
        },
        {
            "code": "eugenia1",
            "iou": 0.6262039443510167
        },
        {
            "code": "laurus234",
            "iou": 0.0501716754754832
        },
        {
            "code": "litsea190",
            "iou": 0.17575950937644014
        },
        {
            "code": "desmodium26",
            "iou": 0.12662864318978598
        },
        {
            "code": "eugenia157",
            "iou": 0.12325939402421104
        },
        {
            "code": "amborella54",
            "iou": 0.16213994648384897
        },
        {
            "code": "ulmus311",
            "iou": 0.27469125065228733
        },
        {
            "code": "eugenia167",
            "iou": 0.15890674118057982
        },
        {
            "code": "ulmus69",
            "iou": 0.15064835337440444
        },
        {
            "code": "eugenia182",
            "iou": 0.6380151066067502
        },
        {
            "code": "convolvulaceae18",
            "iou": 0.321689259645464
        },
        {
            "code": "laurus448",
            "iou": 0.3584210715440727
        },
        {
            "code": "laurus277",
            "iou": 0.6237546225799434
        },
        {
            "code": "convolvulaceae139",
            "iou": 0.0658070644525753
        },
        {
            "code": "litsea139",
            "iou": 0.2809521311016668
        },
        {
            "code": "monimiaceae300",
            "iou": 0.1968038452092762
        },
        {
            "code": "magnolia90",
            "iou": 0.8054095238095238
        },
        {
            "code": "eugenia552",
            "iou": 0.4394192256341789
        },
        {
            "code": "magnolia14",
            "iou": 0.18494798669591678
        },
        {
            "code": "laurus13",
            "iou": 0.3377670716380394
        },
        {
            "code": "magnolia76",
            "iou": 0.34678429642699604
        },
        {
            "code": "rubus96",
            "iou": 0.10375848268661911
        },
        {
            "code": "eugenia566",
            "iou": 0.5400886213527297
        },
        {
            "code": "desmodium338",
            "iou": 0.07205920766216804
        },
        {
            "code": "castanea20",
            "iou": 0.7989216324253974
        },
        {
            "code": "litsea166",
            "iou": 0.14985796692284903
        },
        {
            "code": "monimiaceae69",
            "iou": 0.3758909985491705
        },
        {
            "code": "eugenia543",
            "iou": 0.33521642873501795
        },
        {
            "code": "eugenia124",
            "iou": 0.3299338141557376
        },
        {
            "code": "litsea38",
            "iou": 0.6248545849982298
        },
        {
            "code": "laurus388",
            "iou": 0.5261138350031976
        },
        {
            "code": "eugenia174",
            "iou": 0.1644774713485792
        },
        {
            "code": "rubus102",
            "iou": 0.5485531986343538
        },
        {
            "code": "monimiaceae210",
            "iou": 0.5323784926038976
        },
        {
            "code": "convolvulaceae147",
            "iou": 0.1550494153674833
        },
        {
            "code": "castanea18",
            "iou": 0.1064558998391271
        },
        {
            "code": "monimiaceae70",
            "iou": 0.12215589372217057
        },
        {
            "code": "monimiaceae509",
            "iou": 0.10578478964401294
        },
        {
            "code": "castanea87",
            "iou": 0.744265678502744
        },
        {
            "code": "ulmus486",
            "iou": 0.08755110917790343
        },
        {
            "code": "magnolia165",
            "iou": 0.16701640421144828
        },
        {
            "code": "ulmus104",
            "iou": 0.618138596743985
        },
        {
            "code": "litsea150",
            "iou": 0.16456634544106746
        },
        {
            "code": "ulmus1",
            "iou": 0.12627813645088118
        },
        {
            "code": "ulmus267",
            "iou": 0.6426392572944297
        },
        {
            "code": "ulmus54",
            "iou": 0.43896948914094785
        },
        {
            "code": "ulmus291",
            "iou": 0.533555703802535
        },
        {
            "code": "monimiaceae506",
            "iou": 0.09944877067109983
        },
        {
            "code": "litsea195",
            "iou": 0.6115435865067677
        },
        {
            "code": "litsea36",
            "iou": 0.27945760324029234
        },
        {
            "code": "monimiaceae235",
            "iou": 0.2606392278910375
        },
        {
            "code": "litsea426",
            "iou": 0.356280512522251
        },
        {
            "code": "rubus37",
            "iou": 0.3046476709267407
        },
        {
            "code": "ulmus7",
            "iou": 0.6382198952879581
        },
        {
            "code": "magnolia574",
            "iou": 0.31450759545886614
        },
        {
            "code": "monimiaceae83",
            "iou": 0.13452562018146536
        },
        {
            "code": "rubus119",
            "iou": 0.7686007551986385
        },
        {
            "code": "laurus135",
            "iou": 0.418049104180491
        },
        {
            "code": "magnolia52",
            "iou": 0.44591777750529876
        },
        {
            "code": "rubus101",
            "iou": 0.13849798458383425
        },
        {
            "code": "ulmus593",
            "iou": 0.31084379358437936
        },
        {
            "code": "rubus78",
            "iou": 0.17374110203005536
        },
        {
            "code": "laurus67",
            "iou": 0.2840304450473942
        },
        {
            "code": "litsea120",
            "iou": 0.7640409498080478
        },
        {
            "code": "rubus121",
            "iou": 0.16026356156742436
        },
        {
            "code": "ulmus679",
            "iou": 0.18475170501125046
        },
        {
            "code": "ulmus317",
            "iou": 0.4971468647390736
        },
        {
            "code": "litsea80",
            "iou": 0.691963963963964
        },
        {
            "code": "ulmus97",
            "iou": 0.0378371737947737
        },
        {
            "code": "ulmus273",
            "iou": 0.296488329583802
        },
        {
            "code": "castanea123",
            "iou": 0.2672556598564329
        },
        {
            "code": "laurus4",
            "iou": 0.6902548536014671
        },
        {
            "code": "eugenia16",
            "iou": 0.2415763718979378
        },
        {
            "code": "convolvulaceae154",
            "iou": 0.10255822563867487
        },
        {
            "code": "eugenia63",
            "iou": 0.288001546600116
        },
        {
            "code": "castanea51",
            "iou": 0.6704147465437788
        },
        {
            "code": "monimiaceae196",
            "iou": 0.1369283048298145
        },
        {
            "code": "eugenia275",
            "iou": 0.06289891059156802
        },
        {
            "code": "convolvulaceae463",
            "iou": 0.11700808578264438
        },
        {
            "code": "ulmus300",
            "iou": 0.5782956208488124
        },
        {
            "code": "litsea196",
            "iou": 0.13358057518718075
        },
        {
            "code": "monimiaceae411",
            "iou": 0.02737280571258554
        },
        {
            "code": "desmodium55",
            "iou": 0.2940264446109088
        },
        {
            "code": "monimiaceae175",
            "iou": 0.4293629792750715
        },
        {
            "code": "castanea327",
            "iou": 0.8469704135792737
        },
        {
            "code": "castanea323",
            "iou": 0.42149105991425884
        },
        {
            "code": "litsea121",
            "iou": 0.5274371737282991
        },
        {
            "code": "magnolia344",
            "iou": 0.16552406657669816
        },
        {
            "code": "magnolia172",
            "iou": 0.5566079434236282
        },
        {
            "code": "eugenia215",
            "iou": 0.756479217603912
        },
        {
            "code": "monimiaceae548",
            "iou": 0.405783388336295
        },
        {
            "code": "laurus295",
            "iou": 0.30461629961341335
        },
        {
            "code": "convolvulaceae27",
            "iou": 0.0820254045589003
        },
        {
            "code": "ulmus110",
            "iou": 0.24084677207543206
        },
        {
            "code": "monimiaceae229",
            "iou": 0.18782100394426945
        },
        {
            "code": "laurus285",
            "iou": 0.14447379007573505
        },
        {
            "code": "desmodium34",
            "iou": 0.427073837739289
        },
        {
            "code": "magnolia264",
            "iou": 0.8534848284086392
        },
        {
            "code": "convolvulaceae40",
            "iou": 0.5156162334019077
        },
        {
            "code": "monimiaceae306",
            "iou": 0.13479222237821142
        },
        {
            "code": "laurus192",
            "iou": 0.7882863340563991
        },
        {
            "code": "litsea103",
            "iou": 0.30585436672438826
        },
        {
            "code": "convolvulaceae46",
            "iou": 0.14277486910994763
        },
        {
            "code": "rubus52",
            "iou": 0.22088410064711794
        },
        {
            "code": "convolvulaceae111",
            "iou": 0.22176530290834004
        },
        {
            "code": "amborella103",
            "iou": 0.6800649300456056
        },
        {
            "code": "amborella60",
            "iou": 0.5933781401498458
        },
        {
            "code": "monimiaceae585",
            "iou": 0.3422717575222541
        },
        {
            "code": "eugenia141",
            "iou": 0.21985330244963852
        },
        {
            "code": "convolvulaceae148",
            "iou": 0.06428347444339919
        },
        {
            "code": "litsea41",
            "iou": 0.8501033624435725
        },
        {
            "code": "monimiaceae494",
            "iou": 0.17495085161537258
        },
        {
            "code": "ulmus674",
            "iou": 0.04056745518711346
        },
        {
            "code": "magnolia142",
            "iou": 0.4172469295636656
        },
        {
            "code": "castanea28",
            "iou": 0.35947940473490464
        },
        {
            "code": "litsea91",
            "iou": 0.22883974235891708
        },
        {
            "code": "rubus128",
            "iou": 0.3880960045784598
        },
        {
            "code": "monimiaceae230",
            "iou": 0.360694027360694
        },
        {
            "code": "amborella90",
            "iou": 0.11839484263518163
        },
        {
            "code": "litsea446",
            "iou": 0.0
        },
        {
            "code": "convolvulaceae55",
            "iou": 0.04544470893745942
        },
        {
            "code": "laurus28",
            "iou": 0.192966031336666
        },
        {
            "code": "convolvulaceae103",
            "iou": 0.727533030637706
        },
        {
            "code": "ulmus160",
            "iou": 0.5324349319593077
        },
        {
            "code": "rubus655",
            "iou": 0.5262067536772322
        },
        {
            "code": "ulmus220",
            "iou": 0.5439935964534204
        },
        {
            "code": "ulmus153",
            "iou": 0.17347349978098991
        },
        {
            "code": "laurus76",
            "iou": 0.7719193129618535
        },
        {
            "code": "ulmus115",
            "iou": 0.057045454545454545
        },
        {
            "code": "castanea38",
            "iou": 0.22847179844826082
        },
        {
            "code": "castanea59",
            "iou": 0.4236606906763975
        },
        {
            "code": "castanea83",
            "iou": 0.18652740782279545
        },
        {
            "code": "ulmus74",
            "iou": 0.166102877070619
        },
        {
            "code": "desmodium44",
            "iou": 0.14508007552975732
        },
        {
            "code": "convolvulaceae79",
            "iou": 0.16227324263038548
        },
        {
            "code": "convolvulaceae202",
            "iou": 0.27281437125748503
        },
        {
            "code": "amborella94",
            "iou": 0.22131752954417558
        },
        {
            "code": "desmodium114",
            "iou": 0.09868363085273231
        },
        {
            "code": "eugenia534",
            "iou": 0.15722500917655696
        },
        {
            "code": "monimiaceae122",
            "iou": 0.2783559156306747
        },
        {
            "code": "convolvulaceae24",
            "iou": 0.23649740132041017
        },
        {
            "code": "ulmus20",
            "iou": 0.05356177993132658
        },
        {
            "code": "amborella61",
            "iou": 0.27687586098662526
        },
        {
            "code": "desmodium129",
            "iou": 0.06327643850079183
        },
        {
            "code": "eugenia2",
            "iou": 0.6157014157014157
        },
        {
            "code": "laurus422",
            "iou": 0.3739214942637717
        },
        {
            "code": "laurus128",
            "iou": 0.6144191616766467
        },
        {
            "code": "convolvulaceae23",
            "iou": 0.0790906235823708
        },
        {
            "code": "ulmus191",
            "iou": 0.04271247676781713
        },
        {
            "code": "desmodium24",
            "iou": 0.15590630655714882
        },
        {
            "code": "ulmus35",
            "iou": 0.4273006134969325
        },
        {
            "code": "amborella57",
            "iou": 0.13933796571109328
        },
        {
            "code": "monimiaceae234",
            "iou": 0.30556388854576577
        },
        {
            "code": "ulmus303",
            "iou": 0.25866228453550577
        },
        {
            "code": "ulmus503",
            "iou": 0.021709941234938183
        },
        {
            "code": "amborella81",
            "iou": 0.2342042424136789
        },
        {
            "code": "eugenia177",
            "iou": 0.14556962025316456
        },
        {
            "code": "desmodium61",
            "iou": 0.14242196116878797
        },
        {
            "code": "eugenia113",
            "iou": 0.7621260774040899
        },
        {
            "code": "desmodium19",
            "iou": 0.32709873099986053
        },
        {
            "code": "rubus29",
            "iou": 0.05615780745395828
        },
        {
            "code": "convolvulaceae112",
            "iou": 0.21726311329249132
        },
        {
            "code": "laurus486",
            "iou": 0.28081923419412286
        },
        {
            "code": "eugenia58",
            "iou": 0.235362115771959
        },
        {
            "code": "monimiaceae513",
            "iou": 0.26695083267248215
        },
        {
            "code": "litsea34",
            "iou": 0.8328192796277902
        },
        {
            "code": "amborella92",
            "iou": 0.30753592709428673
        },
        {
            "code": "monimiaceae227",
            "iou": 0.6595893064703603
        },
        {
            "code": "magnolia83",
            "iou": 0.24223307795037605
        },
        {
            "code": "laurus88",
            "iou": 0.1896355353075171
        },
        {
            "code": "monimiaceae238",
            "iou": 0.29250753077606173
        },
        {
            "code": "laurus39",
            "iou": 0.7704837807606264
        },
        {
            "code": "rubus3",
            "iou": 0.519906610960924
        },
        {
            "code": "monimiaceae54",
            "iou": 0.19135780906137562
        },
        {
            "code": "litsea40",
            "iou": 0.259804776015339
        },
        {
            "code": "litsea142",
            "iou": 0.21442892759384208
        },
        {
            "code": "laurus64",
            "iou": 0.39933824952920677
        },
        {
            "code": "ulmus120",
            "iou": 0.38388433108602066
        },
        {
            "code": "magnolia64",
            "iou": 0.11180839655240894
        },
        {
            "code": "monimiaceae195",
            "iou": 0.3695949911954608
        },
        {
            "code": "ulmus62",
            "iou": 0.2884419103931299
        },
        {
            "code": "convolvulaceae153",
            "iou": 0.17530045586406962
        },
        {
            "code": "castanea3",
            "iou": 0.43034691639541156
        },
        {
            "code": "monimiaceae497",
            "iou": 0.2232433196663368
        },
        {
            "code": "eugenia57",
            "iou": 0.23257273345303048
        },
        {
            "code": "rubus554",
            "iou": 0.2319631528634744
        },
        {
            "code": "ulmus172",
            "iou": 0.3920981741993415
        },
        {
            "code": "desmodium698",
            "iou": 0.3869047619047619
        },
        {
            "code": "ulmus141",
            "iou": 0.09173894928222416
        },
        {
            "code": "rubus24",
            "iou": 0.23238502504404981
        },
        {
            "code": "convolvulaceae157",
            "iou": 0.32282013209769084
        },
        {
            "code": "laurus40",
            "iou": 0.1939505201503628
        },
        {
            "code": "laurus57",
            "iou": 0.7685806205354708
        },
        {
            "code": "desmodium38",
            "iou": 0.2807813044994768
        },
        {
            "code": "laurus104",
            "iou": 0.26432914542045594
        },
        {
            "code": "rubus560",
            "iou": 0.5436091278174436
        },
        {
            "code": "convolvulaceae128",
            "iou": 0.36661089421825616
        },
        {
            "code": "castanea175",
            "iou": 0.21522045855379188
        },
        {
            "code": "convolvulaceae127",
            "iou": 0.03955167771152804
        },
        {
            "code": "monimiaceae183",
            "iou": 0.7036328871892925
        },
        {
            "code": "amborella99",
            "iou": 0.20848161328588374
        },
        {
            "code": "litsea263",
            "iou": 0.6312252490099604
        },
        {
            "code": "monimiaceae504",
            "iou": 0.19347290640394088
        },
        {
            "code": "monimiaceae9",
            "iou": 0.23026706231454006
        },
        {
            "code": "magnolia289",
            "iou": 0.18205896703143515
        },
        {
            "code": "convolvulaceae98",
            "iou": 0.07761824176974476
        },
        {
            "code": "eugenia550",
            "iou": 0.2312914492014284
        },
        {
            "code": "convolvulaceae168",
            "iou": 0.1508108108108108
        },
        {
            "code": "ulmus246",
            "iou": 0.5183190606515405
        },
        {
            "code": "litsea10",
            "iou": 0.16697808434471317
        },
        {
            "code": "ulmus263",
            "iou": 0.14909971937041364
        },
        {
            "code": "monimiaceae212",
            "iou": 0.21861294099145537
        },
        {
            "code": "eugenia540",
            "iou": 0.22061131982291832
        },
        {
            "code": "laurus524",
            "iou": 0.7396523576298857
        },
        {
            "code": "rubus76",
            "iou": 0.3302602696770147
        },
        {
            "code": "amborella31",
            "iou": 0.26274812372869466
        },
        {
            "code": "laurus37",
            "iou": 0.37663315702826716
        },
        {
            "code": "ulmus149",
            "iou": 0.18361222904751817
        },
        {
            "code": "desmodium12",
            "iou": 0.2905239531458231
        },
        {
            "code": "magnolia30",
            "iou": 0.8156813847733607
        },
        {
            "code": "desmodium339",
            "iou": 0.6628779755522196
        },
        {
            "code": "laurus103",
            "iou": 0.7353707240840424
        },
        {
            "code": "desmodium91",
            "iou": 0.3173490520949498
        },
        {
            "code": "rubus536",
            "iou": 0.35033124440465535
        },
        {
            "code": "convolvulaceae141",
            "iou": 0.46015959292240083
        },
        {
            "code": "monimiaceae163",
            "iou": 0.3293818650234129
        },
        {
            "code": "convolvulaceae42",
            "iou": 0.4282644021596687
        },
        {
            "code": "litsea27",
            "iou": 0.23570272105290646
        },
        {
            "code": "laurus462",
            "iou": 0.4118419418371511
        },
        {
            "code": "monimiaceae566",
            "iou": 0.574837429736581
        },
        {
            "code": "ulmus690",
            "iou": 0.2823658574839238
        },
        {
            "code": "desmodium484",
            "iou": 0.3184443467888756
        },
        {
            "code": "ulmus200",
            "iou": 0.62380207979338
        },
        {
            "code": "ulmus28",
            "iou": 0.0390279823269514
        },
        {
            "code": "eugenia19",
            "iou": 0.1500182758080518
        },
        {
            "code": "eugenia189",
            "iou": 0.11905881533343783
        },
        {
            "code": "desmodium104",
            "iou": 0.12271599762428816
        },
        {
            "code": "castanea136",
            "iou": 0.7353220596617899
        },
        {
            "code": "monimiaceae201",
            "iou": 0.1562203895420249
        },
        {
            "code": "magnolia5",
            "iou": 0.7368916059252293
        },
        {
            "code": "litsea107",
            "iou": 0.3826539522861434
        },
        {
            "code": "amborella14",
            "iou": 0.20732430734873333
        },
        {
            "code": "eugenia125",
            "iou": 0.6198361718427721
        },
        {
            "code": "amborella5",
            "iou": 0.3756771445100892
        },
        {
            "code": "magnolia91",
            "iou": 0.5284793536250655
        },
        {
            "code": "magnolia12",
            "iou": 0.07216602356867244
        },
        {
            "code": "eugenia555",
            "iou": 0.6626827570201903
        },
        {
            "code": "castanea100",
            "iou": 0.259217779333275
        },
        {
            "code": "laurus405",
            "iou": 0.5331804338835207
        },
        {
            "code": "monimiaceae50",
            "iou": 0.3857662423541081
        },
        {
            "code": "convolvulaceae113",
            "iou": 0.062392935982339956
        },
        {
            "code": "magnolia84",
            "iou": 0.1716434961421516
        },
        {
            "code": "litsea67",
            "iou": 0.2827441018860046
        },
        {
            "code": "eugenia83",
            "iou": 0.18905228471771862
        },
        {
            "code": "monimiaceae205",
            "iou": 0.6301379216168237
        },
        {
            "code": "litsea32",
            "iou": 0.21088398780706474
        },
        {
            "code": "castanea11",
            "iou": 0.6010344569101221
        },
        {
            "code": "amborella96",
            "iou": 0.5693774319066148
        },
        {
            "code": "castanea153",
            "iou": 0.6914176656527703
        },
        {
            "code": "litsea434",
            "iou": 0.05339138405132906
        },
        {
            "code": "rubus70",
            "iou": 0.6500805942953255
        },
        {
            "code": "castanea4",
            "iou": 0.07350032382852842
        },
        {
            "code": "rubus7",
            "iou": 0.19876593701655493
        },
        {
            "code": "rubus89",
            "iou": 0.3066987397374889
        },
        {
            "code": "laurus280",
            "iou": 0.21487258042055424
        },
        {
            "code": "laurus311",
            "iou": 0.361779864445072
        },
        {
            "code": "ulmus686",
            "iou": 0.38018271257905834
        },
        {
            "code": "ulmus195",
            "iou": 0.07870491774619325
        },
        {
            "code": "desmodium3",
            "iou": 0.1460381318873535
        },
        {
            "code": "castanea9",
            "iou": 0.47451820128479655
        },
        {
            "code": "rubus42",
            "iou": 0.21676604700105226
        },
        {
            "code": "laurus278",
            "iou": 0.2522894092974484
        },
        {
            "code": "castanea107",
            "iou": 0.27113100963640496
        },
        {
            "code": "monimiaceae141",
            "iou": 0.2778960594743639
        },
        {
            "code": "monimiaceae278",
            "iou": 0.8376952300548755
        },
        {
            "code": "monimiaceae254",
            "iou": 0.18016899043087312
        },
        {
            "code": "ulmus213",
            "iou": 0.7286108555657773
        },
        {
            "code": "ulmus118",
            "iou": 0.05505166576024982
        }
    ],
    "average_iou": 0.3514578068287064
}
"""

def calculer_moyenne_iou(json_string):
    # Charger les données JSON
    donnees = json.loads(json_string)
    
    # Récupérer la liste des échantillons
    echantillons = donnees.get("samples", [])
    
    # Vérifier que la liste n'est pas vide pour éviter une division par zéro
    if not echantillons:
        return 0.0
    
    # Extraire toutes les valeurs iou
    valeurs_iou = [echantillon["iou"] for echantillon in echantillons]
    
    # Calculer la moyenne
    moyenne = sum(valeurs_iou) / len(valeurs_iou)
    
    return moyenne

# Exécution
moyenne_calculee = calculer_moyenne_iou(json_data)
print(f"La moyenne des IoU calculée est : {moyenne_calculee}")