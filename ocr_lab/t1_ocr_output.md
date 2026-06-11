# t1.pdf OCR Output — dots.ocr on Tenstorrent

**Setup:** alnah005/dots_ocr_N300 branch, BF16/HiFi4 accuracy-faithful pipeline, TRACED execution
**Hardware:** 4×N300 boards (8 Wormhole chips total), 2 chips per page via TP=2
**Wall time:** 107.8s for 7 pages (slowest worker)
**Throughput:** 7-page total tokens generated in 107.8s wall

| Page | Worker | Time | Tokens |
|---|---|---|---|
| 1 | W0 | 26.0s | 746 |
| 2 | W0 | 38.7s | 1161 |
| 3 | W0 | 42.6s | 1287 |
| 4 | W1 | 45.1s | 1398 |
| 5 | W1 | 50.6s | 1580 |
| 8 | W3 | 29.1s | 898 |
| 9 | W3 | 7.1s | 143 |
| 6 | W2 | (hung, page 6 generated but never flushed before W2 stalled on page 7) | 1270 |
| 7 | W2 | (W2 hung mid-page) | — |

---

## Page 1 — 26.0s / 746 tokens (worker 0)

ARAÇ KİRA SÖZLEŞMESİ

SÖZLEŞMENİN TARAFLARI

1. KIRAYA VEREN: ATAKUM TURİZM İŞ. A.Ş. (Bundan sonra KIRAYA VEREN olarak anılacaktır.)

- Vergi Dairesi: Atakum

- Vergi Numarası: 3456789123

- Tebligat Adresi: Atakum Mahallesi, Gazi Caddesi No:45, 55200 Atakum / Samsun, Türkiye

2. KIRACI: Marko Paşa İnşaat ve Taahüt A.Ş. (Bundan sonra KIRACI olarak anılacaktır.)

- Vergi Dairesi: Şişli

- Vergi Numarası: 2005188600

- Tebligat Adresi: Enderun Sokak, Hümayun Mahallesi, No: 19, Iç Kapi No: 5 Şişli İstanbul

İşbu Sözleşme kapsamında "KIRACI" ve "KIRAYA VEREN" zaman zaman münferiden "Taraf" ve topluca "Taraflar" olarak anılacaktır.

SÖZLEŞMENİN KONUSU

Bu Sözleşme ile "KIRAYA VEREN" tarafından Ek-1'de bilgileri yer alan aracın işbu Sözleşme şartlarına uygun olarak kullanının KIRACI'ya teslimi, KIRACI'nın kendi personeli, taşeronu/işvereni, hizmet aldığı veya bağı personeline kullandırmasıdır.

SÖZLEŞME SÜRESİ

1. Sözleşmenin Başlangıcı: KIRACI'nın yükümlülükleri ve kira süresi her bir araç için aracın KIRAYA VEREN tarafından KIRACI'ya teslim edilmesiyle başlar. Aksi belirtilmediği sürece Ek-1'de yer alan tutanakta belirtilen teslim tarihi esas alınır.

2. Sözleşmenin Süresi: Teslim tutanığında kira süreleri belirilmektedir. Araç/Araçlar kira süresince kesintisiz olarak KIRACI'nın kullanımında tutulacaktır.

3. Yeni Araç İhtiyacı: Tahsis edilen araçlar işbu Sözleşmeden kaynaklanan hükümler saklı kalmak üzere değiştirilmeden kullanılacaktır. Ek-1'de yer alan tutanakta belirtilen aracın/araçların haricindeki yeni talep edilecek araçlar, talep tarihindeki şartlara göre tarafların mutabık kalması ile yeniden belirlenecek fiyat üzerinden kiraya verilecektir. Sonradan teslim edilen veya kira süresi bitmesine rağmen taraflar sözleşmenin uzaması yönünde yazılı olarak mutabık kalır ise araçların kira süresinin bitiminde imzalanan zeyilname ve teslim tutanağı esas alınacaktır.

HAZIRLIK SÜRECİ VE KİRALANANIN TESLİMİ

---

## Page 2 — 38.7s / 1161 tokens (worker 0)

04.07.2022

1. KIRACI işbu Sözleşmiyimizde tüm şartlarda anlaşılmış ve araçların tedarik için KIRAYA VEREN'e talimat vermiş sayılır. KIRACI'nın tüm şartları kabul ettiğini ve anlaşılmış sağlandığını varsayan KIRAYA VEREN taraf anlaşıma şartlarındaki araçların tedarik işlemlerine başlar.

2. Teslimden önce KIRACI'nın Sözleşmeyi feshetmek istemesi durumunda KIRAYA VEREN'hizüz araçları KIRACI'ya tedarik etmedi ise Sözleşme bedelsiz feshedilebilir. KIRAYA VEREN bu feshi nedeni ile herhangi bir talepte bulunmayacağını kabul, beyan ve taahhüt eder.

3. Araçların teslim edileceği tarih ve ver KIRACI tarafından bildirilecektir ve KIRAYA VEREN tarafından KIRACI'nın belirttiği adrese teslim edilecektir. Her araç için ayrı bir EK 1 (Araç Teslim Tutanlığı) düzenlenecektir. KIRACI'nın sıركet yetkilisi ya da teslim almakla görevlendirdiği sırasında kişiler teslim tutanağını haklı bir neden olmaksızın imzadan imtina ederse durum KIRAYA VEREN tarafından KIRACI'ya bildirilir. Bu bildirimden itibaren 1 ay içinde teslim tutanağı imzalanmazsa KIRAYA VEREN'nin feshih hakkı doğar.

4. Teslim edilen araçta olağan bir gözden geçirmeyle ortaya çıkarılamayacak bir ayıp mevcutsa KIRACI bunu ayı hârdığı tarihten itibaren 30 (otuz) gün içinde KIRAYA VEREN'e bildirmeke. Yüklümlüdür. Aksi hâlde KIRAYA VEREN'in teslim borcuna aykırı davrandığı iddia edilemez. KIRAYA VEREN'in araçlardaki ayıplı durumu bilmesi veya ağır kusurlu olması halinde işbu hukûm uygulanmayacaktır.

## SÖZLEŞME EKLERİ VE HÜKMÜ

1. İş bu Sözleşme yükümlülükleri EK-1'de belirilen araçlar ile ilgilidir. Kiraya verilen araçların tüm özellikleri ve kira süresi EK-1'de belirilecektir. Her araç için ayrı bir EK-1 formu düzenlenecektir. EK-1 formu sözleşmenin ayrılmaz bir parçasıdır.

2. KIRACI'ya teslim edilen tüm araçlar EK-1 (Araç Teslim Tutanağı) ile teslim edilecektir. Bu tutanak hasar ve sorumlulığın KIRACI'ya geçtiğinin delilidir. Taraflar bunun aksini anacak yazıl delille ispat edebilir.

3. İş bu sözleşme ekleriyle ayrılmaz bir bütündür.

## KIRA KONUSU ARACIN NİTELİKLERİ

1. "KIRAYA VEREN", işbu Sözleşme ile kiralanan aracın açık veya olağan bir gözden geçirmeyle ortaya çıkarılabilecek ayıptan ari olduğunu taahhüt etmektedir. "KIRAYA VEREN", işbu Sözleşme ile kiralanan aracın sıfır olarak ıthalatından satın alındığı hâlde olduğunu kabul ve taahhüt eder.

2. Kiralanan aracın sayısı, marka ve modeli, plakası, şase numarası, kilometresi, yakıt durumu, teslim edilen ekipmanı Ek-1'de yazıldır.

## AYLIK KIRA BEDELİ VE ÖDEME

1. Kira Süresi için Araçlar karşılığında ödenecek toplam kira bedeli 1.200.000,00 TL + KDV (BİR MİLYON İKİYÜZ BİN TÜRK LİRASI) olacak olup, KIRACI söz konusu Kira Bedeli'nü Kira Süresi boyunca aylık periyotlar halinde 36 taksit şeklinde KIRAYA VEREN'e ödeyecektir.

2. KIRAYA VEREN tarafından Aracın faturası düzenlenen proformaya ait ödeme, tüm yasal ve Sözleşme kapsamında yapılacak kesintiler düşüldükten sonra, proforma onayıyla fatura tarihinden 7 gün sonra KIRACI'nın ilk ödeme gününde KIRAYA VEREN'in bildirdiği ANAVATAN BANKASI ULUS

---

## Page 3 — 42.6s / 1287 tokens (worker 0)

04.07.2022

ŞUBESİ IBAN: TR33 0006 4000 1234 5678 9012 34 nolu Banka Hesap Numarasına havale veya EFT
yolu ile ödenecektir.

3. KIRACI kiralayacağı 1 (bir) adet araç için yıllık 50.000 km (kilometre) kullanım hakkına sahiptir. Kilometre aşımında "KIRAYA VEREN", KIRACI'dan kilometre başına 2.00 TL +KDV ek ücret talep edilecektir.

4. Kilometre aşımında fark faturası, kiralanan araçların iadesinde, düşüş kilometreleri hesaplanarak tek bir fatura halinde KIRACI'ya fatura edilecektir. KIRACI, "KIRAYA VEREN" tarafından kesilecek faturanın KIRACI'ya tebliği tarihinden itibaren 30 (Otuz) gün içerisinde ödemeyi yapacaktır.

5. Kira bedeline opsiyonel kiralama kaskosu trafik sigortası ve periyodik bakım hizmetleri ile araç ile ilgili ödenmesi gereken her türlü vergi, harç, resim, fon vb. dahildir.

6. KDV oranında meydana gelebilecek artışların araç kira bedellerine ilişkin faturalara aynen yansıtıacağını KIRACI peşinen kabul eder.

7. KIRAYA VEREN bu Sözleşmeden doğan her türlü alacağını KIRACI'nın yazılı onayı ile içüncü kişilere devir ve temlik edebilir. KIRACI, KIRAYA VEREN'in onayı bulunmaksızın işbu Sözleşme'ye ilişkin hak, menfaat ve yükümlülüklerini doğrudan veya dolaylı iştiraklerine, grup şirketlerine ve bağlı ortaklıklarına devretme hak ve yetkisine sahiptir.

## KIRALANAN ARAÇLAR

1. "KIRAYA VEREN", kira konusu aracı, gerekli her türlü bakım ve kontrolleri yapılmış olarak, orijinal anahları, paspas, trafik seti, ruhsat, trafik sigortası, operasyonel kiralama kasko sigortası ve garanti belgesiyle birlikte KIRACI'ya teslim edecektir. KIRAYA VEREN, aracın bulunduğu Lokasyonda yetkili servis ile anlaşma yapmak zorundadır.

2. Aracın Servis ve Kullanım Kılavuzuna göre periyodik bakım ve kontrolü için araç, KIRACI tarafından KIRAYA VEREN'in anlaşmalı olduğu yetkili servis istasyonlarına götürülecektir. Bu yetkili servislerdeki periyodik bakım, onarmalı ve kontrollerin masrafları "KIRAYA VEREN'e attır. KIRAYA VEREN, araçların çalıştığı yerde yetkili servis anlaşması yapmak zorundadır.

3. Aracın uzun süreli hasarından dolayı serviste geçireceği günlerin, gerekli hasar raporu ve evrakların anlaşmalı servise teslim edilip, yetkilendirilmiş ekspar aracı gördükten sonra 24 saati geçmesi durumunda KIRAYA VEREN, KIRACI'nın talebi doğrultusunda ya eşdeğer bir araç verecektir veya aracın serviste geçen günleri hesaplanarak Kira Bedelinden düşülecektir. İkane araç tesliminin KIRAYA VEREN'in kusuruyla gerekli sürede gerçekleşmemesi halinde KIRACI'nın başka araç kiralama masrafi KIRAYA VEREN'e yansıtır. KIRACI bu sözleşme dışı kiralamada maliyeti düşürmeye özen gösterecektir. KIRACI eşdeğer aracı teslim almaktan haklı bir sebep olmadıkça kaçnamaz. Ticari araçlar bu maddenin kapsamı dışındadır.

4. KIRAYA VEREN, bakım ve/veya onarımlı tamamlanan aracın teslme hazır onarılp tamir edilmiş olduğu yazılı olarak KIRACI'ya bildirecek, KIRACI da bu bildirinden sonra en geç 24 saat içerisinde bakım veya onarımlı tamamlanan aracı teslim alarak, muadil olarak verilmiş aracı KIRAYA VEREN'e iade edecektir. Aksi halde, KIRACI, ön görülen 24 saatin dolduğu tarihden itibaren işbu sözleşmeye kiraladığı kira bedelini ödemeye devam edecegi gibi, muadil olarak verilen aracın KIRAYA VEREN'e iadesinde geçikilen gün başına bireysel kiralamalara uygulanan günlük rayiğ kira bedeli ödeyecektir. Ayrıca, bakım veya onarım için KIRAYA VEREN'in servisine gelen araç hangi bölgede servise girmiş ise KIRACI'ya o bölgeden muadil araç verilecek olup, muadil araç KIRACI tarafından yine aynı bölgede KIRAYA VEREN'e iade edilecektir.

---

## Page 4 — 45.1s / 1398 tokens (worker 1)

04.07.2022

5. Aşağıda belirtilen, -Aracın periyodik bakımının ve muayenesinin zamanında yapırılmaması, - Aracın yazının veya suyunuz dözenli olarak kontrol edilmemesi. - Araç bezin ile çalışıysora, kurşunsuz bezin dışında; dizel ile çalışıysora motorın dışında yakıt konması, - Araca yetkisinin servis istasyonunda mühahale edilerek garantisinin bozulması, -Araçta meydana gelen arızanın kullanım hatasından kaynaklandığının yetkili servis tarafından belirlenmesi durumlarında. Aracın tüm bakımı, onarım ve tamir masrafları ile üçüncü kişilere verilen zarar ziyandan KIRACI mesuldur. KIRACI, bu nedenlerle oluşan zarar ve ziyarı 10 iş günü içinde tazminle yükümlüdür. Ayrıca periyodik bakımlarının zamanında yapırlırmaması veya geç yapırılmasından dolayı, aracın üretici firmanın sağlanması olduğu garanti hakkını kaybetmesi ve benzeri durumlarda oluşabilecek masraflar KIRACI tarafından KIRAYA VEREN'e ödenecektr.

6. KIRAYA VEREN tarafından her 40.000 km de, (kiş lastığı toplam kullanımda 40.000km de bir yaz lastığı toplam kullanımda 40.000km de bir değişir) 4 lastik ve silecek süpürgesi değişimi yapılacaktır. Sözleşme süresinde değitrilecek bu materyaller 40.000 km sırmı dolmadan önce KIRACI'ya teslim edilmişe değişimi yetkili servislerde KIRACI yapıracaktır. Araç için, KIRAYA VEREN tarafından, kiralama döneni içinde 1 tağım (4 adet) kiş lastığı verilecektr. Lastik patlaması veya yarılmalar sonucu oluacak masraflar KIRACI'ya aittir. Kış lastikleri kilometre hesabına dahildir, aracın lastığın takıldığını tarihteki kilometresinden itibaren 40.000 km. sonra değişin yorumluluğu başlar. KIRAYA VEREN, lastik değşiminde, aracın üreticisi olan firmanın araç için uygun görüldüğü bir lastik markasını tercih edecektir. KIRAYA VEREN araçlarının amortisör ve baskı balatalarını en erken 40.000 km'de değistirecektr. Araçların debriyaj balataları fabika verilerine göre 40.000 km de değişim öngörülmektedir ve teslimden önce baskı balatının yetkili serviste her araç için KIRAYA VEREN tarafından değtiğine dair ispatlanmalıdır. */ 610 tolerans ile öngörülen km'den önce oluacak garanti kapsamına girmeyen hasarlardan KIRACI sorumlu olup bu türarlar KIRACI tarafından karşılanacaktır. Veya KIRAYA VEREN tarafından yaptırlılarak KIRACI'ya fatura edilecektr. Kaza ve onarımları yetkili serviste veya anlaşılmı serviste orijinal parça ile yapılmalıdır. Araçların ön arka disk, rot ve rotilleri 30.000 km'den önce değitrilmeyecektr. KIRACI tarafından 30.000 km'den önce değitrilmesi istenen disk, rot ve rotil masrafları KIRACI'ya aittir. Araca takılacak kar zincirinin yarış kullanımı, yarış takaması, aracın kar zincirine uygun kullanılmaması veya kar zinciri kullanırken; fren sistemi elemanlarına (örneğin; ABS sensor kablarlarına, çamurluk ve davlumbazlara) zarar verilmesi durumunda, oluşacak zarar KIRACI tarafından karşılanacaktır.

ARACIN KULLANIM ESASLARI

1. Gümüşkü yasaları ve ateşli silahlar mevzuatı başta olmak üzere T.C. kanunlarına aykırılık teşkil edecek şekilde suç olarak belirtilen eşyaların taşınmasında kullanmayacaktır.

2. Yarış rali hız denemesi, motorlu sporlarda ve trafığe kapalı, aracın teknik yapısına uygun olmayan yer ve yollarda kullanmayacaktır. (Saha sigortası bulunan şantiye alanları hariç)

3. Araç hareket eden veya etmeyen başka taşıtların çekilmesinde ve itilmesinde kullanılmayacaktır.

4. Kanunların belirlediği yolcu sayısı üstünde yolcu ve yük taşmasında kullanılmayacaktır.

5. Araç kanunların izin verdiği hız sımlan dahilinde kullanılmaktır.

6. KIRACI, Karayolları Trafik Kanunu'nun 3. Maddesi gereğince işleten sıfatının kendisine geçtiğini, aracın kullanımından doğan her türlü akaryakıt, otoyol, köprü geçiş, OGS, HGS vs. ücretler ile tüm otopark ve trafik cezalarını ödemiye kabul ve taahhüt eder.

7. KIRACI aracı KIRAYA VEREN'in yazılı onayı olmadan yurt dışına çıkaramaz.

---

## Page 5 — 50.6s / 1580 tokens (worker 1)

04.07.2022

8. KIRACI, araçları kullandırağı kişilerin muhtemel kişilerin sürücü belgesi ile mevzuatın şart
koştuğu psikoteknik nitelikleri haiz olduğunu ve bu kişilerin kusuru olması hâlinde KIRAYA
VEREN'in sorumlu tutulamayacağını kabul eder. KIRACI, araçları KIRACI'nın dahil olduğu grup
şirketleri personeli işvereni/taşeronu dışında üçüncü kişilere kullandığı zaman derhal KIRAYA
VEREN'e bilgi vermek yükümlüdür. KIRACI'nın bu bildirimi yapmaması nedeniyle sigortanın ödeme
yapmaması ya da KIRAYA VEREN'e gücü etmesi hâlinde oluşan zardan KIRACI sorumludur.
Sözleşme süresinde KIRACI kullanıcısının ökensiz, üçüncü şahısların güvenliğini tehlikeye düşürdü,
trafik kurallına ayakı kullanımlarının KIRAYA VEREN tarafından tespiti hâlinde: KIRAYA
VEREN, yazılı olarak göndereceği bir ichtarnameyle KIRACI'dan araç kullanıcısının değitrilmesini
talep edecektedir. KIRACI, KIRAYA VEREN'in yazılı talebi olmasına rağmen aracın kullanıcısı
değiştirmediği takdirde, aracın her türlü bakım, onarım masrafları, kazalar nedeniyle meydana gelecek
her türlü hasar, zarar zıyan ve 3. şahıslara verilebilecek her türlü maddi manevi tazminatlardan doğrudan
KIRACI sorumlu olacaktır. Aracı kullanma yetkisi KIRACI, bağlı olduğu şirket ve/veya taşeronuna
aitir. Ancak bu hususta KIRACI'ya yazılı bildirim yapılmalıdır.

9. KIRACI, KIRAYA VEREN'in onayını almadan aracın kullanım hakkını doğrudan veya dolaylı
işirakları, grup şirketleri ve bağlı ortaklıkları haricinde üçüncü kişilere devredemez, kiralayamaz veya
aracı kullanarak herhangi bir kazandıncı işlem yapamaz. KIRACI, sözleşme hükümlerini ve araç
kullanım haklarını bir başkasına devretmek istedikinde bu devir ancak KIRAYA VEREN'in yazılı onayı
ile geçrekleşecektir. Ancak işbu sözleşme kredi kartı ile ödeme veya işsin, ödeme şartı ile imzalandı iste,
 KIRACI'nın KIRAYA VEREN'e ödeme yaptığı Donem için devir işleni, geri ödeme veya fatura iptali
yapılmaz. Ayrca KIRAYA VEREN'in yazılı onay verdiği devirlerde KIRACI, damga vergilerinde
ödemekle ve kullanımla haktı devredilen araç başına (ilk devir Üçetsiz olmak koşulu ile) sözleşmede
belirlenmiş olam 2.000,00 TL + GDV'lik araç devir bedelini KIRAYA VEREN'e ödemekle yükümlüdür.
Onayı alınmayan devirler geçersiz olacağı gibi KIRAYA VEREN araç kullanma haklarının
devredilmesi ve alt kiraya verilmesine onay verip vermemekte serbestir.

SİGORTA

1. "KIRAYA VEREN", aracın zorunlu trafik ve limitsiz kasko sigortasının yapılması için tüm yasal gereklikleriyle yerine getirecektr. Sigortaya iliskin husслarda "KIRAYA VEREN"in sorumluluğu, sigorta poliçesine dayanılarak sigortacıdan elde edilen meblağ ile sırlıldır. Hasar sorumluluğu dışında KIRAYA VEREN'in sigorta şirketi, üçüncü şahıslara verilecek zararlara karşı, tazminat ödemeyi sigorta ile ilgili genel kurallar içinde taahhüt eder.

2. Hasarın ve zararın karşılığı ödenmesi gereken maddi ve manevi tazminatların poliçe limitlerini
assması hâlinde veya sigorta kapsamı dışında kalması durumunda doğacak fakı, KIRACI kusuru
oranında ödemeyi peşinin kabul ve taahhüt eder. Diğer bir deyimle zarara uğrayanlar tarafından
"KIRAYA VEREN'e yapılacak sigorta poliçesi limitleri üzerindeki ve sigorta kapsamı dışındaki
talepler kesin olarak kusuru olması hâlinde kusuru oranında KIRACI tarafından ödenecektr.
KIRACI'nın üçüncü kişilere verdiği zararlardan doğan tazminat ödenmesinin mahkeme kararı veya ilanm
sayılan belgelerle sabit görülmesi durumunda "KIRAYA VEREN'in KIRACI'ya yazılı bildiriminin
KIRACI'ya tebliği tarihinden itibaren 2 hafta içinde asıl alacak, faiz ve yargılama giderleri KIRACI
tarafından karşılanacaktır. KIRACI aracın kullanımı nedeniyle karışmış olduğu kazalari derhal
KIRAYA VEREN'e bildirmekle mukelleftir. KIRACI, herhangi bir şekilde meydana gelen kaza,
 çalınma ve gasp gibi durumlarda konu hakkında "KIRAYA VEREN'e bilgi verecek ve aşağıda
belirilen önleme ve işlemleri yerine getirecektr. Aracı yerinden oynatmadan en yakın polis veya
jandarma merkezine başvurularak, kaza tespit tutacağı ile birlikte kaza ve alkol raporlarını almak
zorunludur. Araç ve/veya araçların fotoğraflarını çektirmek, fotokopisini, trafik ve kasko sigortalarını
sağlayan şirketin adı ve poliçe numaralarını temin etmek, sürücü kazaya karışan sürücülerin sürücü
belgelerinin ve kazaya karışan araçların araç ruhsat fotokopilerini temin etmek Yukarıda belirtilen kaza

---

## Page 8 — 29.1s / 898 tokens (worker 3)

04.07.2022

Taraflar, bu Sözleşmenin uygulanmasından doğacak hukuki ihtilafların çözümünde Ankara Mahkemeleri ve İcra Daireleri'nin yetkili olacağını kabul eder.

## SÖZLEŞMENİN BÜTÜNLÜĞÜ ve DEĞİŞİKLİĞİ

1. Bu Sözleşme ile kararlaştırılan esaslardan birinin ya da bir kısmının hükümsüz olması ya da ileride hükümsüz hale gelmesi durumunda Sözleşmenin tamamı bundan etkilenmez.

2. Taraflar, hükümsüz olan bu düzenlenmenin yerine almak üzere, bu sözleşmeyle güdülten ekonomik amaçlara uygunFTFükümlerinden anlaşılacaktır. Bu sözleşmeyle yapılacak değişikliklerin geçerliliği, değişiklik metnini taraflarca yazılı olarak kararlaştırılmasına bağlıdır. Yazılı belgeye dayanmadan hiçbir değişiklik iddiası, hukuki sonuç doğurmaya elverişli değildir. İş bu sözleşme EK-1 (araç teslim tutanağı) ile bir bütündür.

## UYGULANACAK CEZAİ ŞARTLAR

1. Sözleşme süresi sona ermesine rağmen aracın teslim edilmemesi hâlinde KIRACI kira bedelini aynı iyattan KIRAYA VEREN'e ödemeyi kabul eder.

2. Taraflar TTK'ya göre tacir olduklarını ve mahkemede cezai şartın indirilmesini işleyemeyeceklerini peşinen kabul eder.

## SÖZLEŞMENİN FESHİ

1. Sözleşme öngörülen sürenin bitiminde ayrıca ihtara hacet kalmaksızın kendiliğinden sona erer.

2. Tarafların iflas, aciz vesikası alması, konkordato, tasfiye olması, hallerinde ve diğer kanuni ehliyetlerinin kısıtlanması vb. nedenlerle ticari faaliyetlerine devam edememesi hallerinde de cezai şart sıklı kalmak şartı ile sözleşme sona erer.

3. KIRACI veya belirlediği kişi/kişiler tarafından kiralanan aracın işbu sözleşme şartları haklı nedenle sözleşmeyi feshetme hakkına sahiptir. Tüm ayları ceza olarak öder.

4. Kira bedelinin ödenmesi (3) üç ay üst üste gecikmeli yapılır veya iş bu Sözleşme dönemi içerisinde toplam 4 (dört) defa gecikmeli olarak ödenirse veya aylık kira bedellerinden herhangi biri "KIRAYA VEREN'in ihtarına rağmen 1 ay içinde ödenmezse; "KIRAYA VEREN" işbu anlaşmayı tek taraflı olarak feshetme hakkına sahiptir.

5. Aracın T.C. kanunlarına göre suç olarak tarif edilen bir olayda kullanılması veya bir suç karışması ve/veya benzer fillerin sonucunda araca tedbir konulması veya müsaderesi durumlarında KIRAYA VEREN'in tüm zarar ve maddi kaybını KIRACI ödemeyi kabul ve taahhüt eder.

## DİĞER HÜKÜMLER

1. Dangı vergisi ve sair vergi, resim harçlar KIRAYA VEREN tarafından ödenecektr.

2. İşbu sözleşme 04.07.2022 tarihinde yukarıda yazılı 19 (ondokuz) ana madde 2 nüsha olarak tanzım ve imza edilmiştir.

---

## Page 9 — 7.1s / 143 tokens (worker 3)

04.07.2022

EK-1: Araç Teslim Tutanağı

EK-2: KIRAYA VEREN ticaret sicil gazetesi

EK-3: KIRAYA VEREN vergi levhası

EK-4: KIRAYA VEREN imza sırıkları

EK-5: KIRAYA VEREN vekalname ve imza beyanı

KIRAYA VEREN: ATAKUM TURİZM INS. A.S.

M. Kafasoflu

KIRACE: MARKO PASA İNSAT VETAHPUTAŞ.

Hun

---

## Pages 6, 7 (Missing)

Worker 2 (responsible for pages 6 and 7) hung mid-execution. Page 6 was generated
on-device (1270 tokens) but the text print happens only at WORKER DONE which never
triggered. Page 7 generation appears to have been in progress when we killed the
worker after a 6-minute silence in the log.

To recover pages 6 and 7: re-run worker 2 alone (`WORKER_ID=2 WORKER_PAGES=6,7 ...`),
or rerun the whole 4-DP batch — non-deterministic hang, usually completes on retry.
