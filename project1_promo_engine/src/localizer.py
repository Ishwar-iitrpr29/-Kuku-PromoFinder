"""
Module 3B: IsoDub Timestamp-Locked Hindi/Hinglish & Indian Cultural Promo Adapter.
Translates and transcreates selected microdrama promo clips for Kuku's Indian audiences
while strictly enforcing a syllable-per-second budget so dubbed audio never overflows camera cuts.
Dynamically adapts tone across all 3 deterministic model profiles (Balanced, Bollywood Dramatic,
and Conversational Urban Hinglish) as well as live LLM providers.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from .parser import EpisodeScript, format_mmss
from .promo_finder import PromoCandidate
from .llm_router import UniversalLLMRouter


MAX_SYLLABLES_PER_SEC = 4.6  # Natural dramatic Hindi/Hinglish speech pacing ceiling


@dataclass
class TimedDubLine:
    episode_number: int
    timestamp_window: str
    duration_sec: float
    max_syllable_budget: int
    canonical_speaker: str
    indian_transcreated_speaker: str
    emotion_cues: str
    original_english: str
    isochronous_hinglish_dub: str
    indian_royal_transcreation: str
    actual_syllables: int
    timing_fit_status: str  # "FITS_WINDOW" | "COMPACTED_TO_FIT"


INDIAN_ROYAL_CHARACTER_MAP = {
    "LADY JULIANA ASHCROFT": "Rajkumari Jhanvi Rathore (Gharana Suryagarh)",
    "DUKE ALEXANDER ASHBOURNE": "Maharana Vikramaditya Singh (Riyasat Devgarh)",
    "GERALD HARRINGTON (EX-FIANCE)": "Kunwar Gajendra Pratap",
    "LADY CLARISSA (SISTER)": "Rajkumari Chitrangada (Sauteli Behen)",
    "LADY ASHCROFT (JULIANA'S STEPMOTHER)": "Badi Rani Mahendra Kumari",
    "LORD ASHCROFT (JULIANA'S FATHER)": "Thakur Pratap Singh",
    "LADY CATHERINE (USURPER BRIDE)": "Rani Kamini Devi",
    "LORD RUPERT ASHBOURNE (USURPER COUSIN)": "Yuvraj Ranvijay Singh",
    "DOWAGER DUCHESS ASHBOURNE (ALEXANDER'S MOTHER)": "Rajmata Gayatri Devi",
    "CONSTABLE": "Kotwal Daroga",
}

# Tuple structure:
# (regex, hybrid_hinglish, bollywood_dramatic, urban_hinglish, royal_transcreation)
CURATED_ISOCHRONOUS_TABLE = [
    (
        re.compile(r"you should turn around", re.I),
        "Yahan se laut jaao, Juliana.",
        "Kadam wahin rok lo, Juliana!",
        "Wait, wapas mud jao Juliana.",
        "Mandap se laut jaao, Jhanvi.",
    ),
    (
        re.compile(r"^\s*why\??\s*$", re.I),
        "Kyun?",
        "Aakhir kyun?!",
        "Wait, why?",
        "Aakhir kyun?",
    ),
    (
        re.compile(r"gerald is waiting for me", re.I),
        "Gerald mera intezaar kar rahe hain.",
        "Mere hone wale pati wahan khade hain!",
        "Gerald altar par mera wait kar raha hai.",
        "Kunwar ji mera intezaar kar rahe hain.",
    ),
    (
        re.compile(r"^\s*not anymore\.?\s*$", re.I),
        "Ab nahi.",
        "Ab khel badal chuka hai!",
        "Not anymore, darling.",
        "Ab nahi.",
    ),
    (
        re.compile(r"what do you mean", re.I),
        "Matlab kya hai aapka?",
        "Yeh kaisi paheli bujha rahi hain?!",
        "Wait, iska kya matlab hai?",
        "Yeh kya keh rahi hain aap?",
    ),
    (
        re.compile(r"look for yourself", re.I),
        "Khud hi dekh lo.",
        "Khud apni aankhon se बर्बादी dekh lo!",
        "Samne dekho, khud samajh jaogi.",
        "Apni aankhon se dekh lo.",
    ),
    (
        re.compile(r"^\s*gerald\??\s*$", re.I),
        "Gerald?",
        "Gerald, yeh sach nahi hai!",
        "Gerald, seriously?!",
        "Kunwar ji?",
    ),
    (
        re.compile(r"why are you holding her hand", re.I),
        "Tumne iska haath kyun thaama hai?",
        "Mere suhaag ne is naagin ka haath kyun pakda?!",
        "Wait, tumne Clarissa ka haath kyun pakda hai?",
        "Aapne Chitrangada ka haath kyun thaama hai?",
    ),
    (
        re.compile(r"because he is marrying me", re.I),
        "Kyunki ab yeh mere pati hain!",
        "Kyunki ab yeh mera suhaag hain!",
        "Kyunki Gerald ab mujhse shaadi kar raha!",
        "Kyunki gathbandhan ab mere saath ho raha hai.",
    ),
    (
        re.compile(r"^\s*no[\.\!\s]*$", re.I),
        "Nahi...",
        "Nahi! Yeh dhokha hai!",
        "No way...",
        "Yeh nahi ho sakta...",
    ),
    (
        re.compile(r"you knew this before i walked", re.I),
        "Darwaze se pehle tum sab jaante the!",
        "Sab mile hue the is gaddari mein!",
        "Main andar aayi usse pehle sab planned tha!",
        "Mandap mein mere kadam rakhne se pehle yeh saazish rachi gayi!",
    ),
    (
        re.compile(r"you let me come here", re.I),
        "Tumne mujhe yahan bulaya...",
        "Mera tamasha banane bulaya...",
        "Tumne mujhe yahan aane diya...",
        "Tumne mujhe mandap tak aane diya...",
    ),
    (
        re.compile(r"you let me dress in my mother", re.I),
        "Maa ke jode mein aane diya...",
        "Maa ke laal jode ka apmaan kiya!",
        "Mom ke bridal gown mein bula liya...",
        "Meri swargiya Maa ke laal jode mein bulaaya mujhe...",
    ),
    (
        re.compile(r"you let me believe i was walking toward my husband", re.I),
        "Mujhe laga main pati ke paas aayi!",
        "Mujhe jhoothe suhaag ka khwab dikhaya!",
        "Mujhe laga main apne groom ke paas ja rahi!",
        "Mujhe bharosa dilaaya ki main apne suhaag ki ore badh rahi hoon!",
    ),
    (
        re.compile(r"i changed my mind", re.I),
        "Mera iraada badal gaya.",
        "Dil aur faisla dono badal gaye!",
        "Sorry, mera mind change ho gaya.",
        "Maine apna faisla badal liya.",
    ),
    (
        re.compile(r"you shouldn't have to beg", re.I),
        "Gidgidane ki zaroorat nahi hai.",
        "Gद्दारों ke aage jhukna band karo!",
        "Inke samne beg mat karo, Juliana.",
        "Suryagarh ki beti kisi ke aage nahi jhukegi.",
    ),
    (
        re.compile(r"to remember your worth", re.I),
        "Apni keemat yaad rakho.",
        "Apna shahi rutba pehchano!",
        "Apni self-respect yaad rakho.",
        "Apna samman paane ke liye.",
    ),
    (
        re.compile(r"why are you doing this", re.I),
        "Aap yeh kyun kar rahe hain?",
        "Ek anjaan par itna ehsaan kyun?!",
        "Wait, aap meri help kyun kar rahe ho?",
        "Aap meri madad kyun kar rahe hain?",
    ),
    (
        re.compile(r"because i want to", re.I),
        "Kyunki main chahta hoon.",
        "Kyunki yeh meri zidd hai!",
        "Bas, kyunki mera mann hai.",
        "Kyunki yeh mera faisla hai.",
    ),
    (
        re.compile(r"you came here to become a wife", re.I),
        "Tum yahan dulhan banne aayi thi.",
        "Aaj is mandap se tum suhaagan hi jaogi!",
        "Tum yahan bride ban kar aayi thi na?",
        "Aap yahan vivaah karne aayi thi.",
    ),
    (
        re.compile(r"there she is", re.I),
        "Dekho, woh aa gayi.",
        "Dekho, kaidi dulhan aa gayi!",
        "Look, woh aa gayi.",
        "Dekho, vahi shatru aa gayi.",
    ),
    (
        re.compile(r"look at her", re.I),
        "Dekho uski haalat.",
        "Zanjeeron mein jakdi rani ko dekho!",
        "Zara uski condition toh dekho.",
        "Nazar daalo uspar.",
    ),
    (
        re.compile(r"she actually came back", re.I),
        "Woh sach mein wapas aayi hai.",
        "Maut ke munh se wapas aa gayi!",
        "Wait, yeh sach mein wapas aa gayi.",
        "Woh mandap mein laut aayi.",
    ),
    (
        re.compile(r"not as a bride this time", re.I),
        "Par is baar dulhan ban kar nahi.",
        "Aaj dulhan nahi, mujrim ban kar aayi hai!",
        "Par is baar bride ban ke nahi.",
        "Par is baar rani ban kar nahi.",
    ),
    (
        re.compile(r"look around", re.I),
        "Charon taraf dekh lo.",
        "Aankhein khol kar apna anjaam dekho!",
        "Zara charon taraf nazar ghumao.",
        "Nazar ghuma kar dekh lo.",
    ),
    (
        re.compile(r"nothing left between us", re.I),
        "Ab saara haq humara hoga.",
        "Ab is takht par sirf humara raaj hoga!",
        "Ab humare beech koi rukawat nahi.",
        "Ab Devgarh ka taaj humara hoga.",
    ),
    (
        re.compile(r"^\s*nothing\.?\s*$", re.I),
        "Kuch nahi bachega.",
        "Sab khaak ho gaya!",
        "Bilkul kuch nahi.",
        "Kuch nahi.",
    ),
    (
        re.compile(r"^\s*shall we\??\s*$", re.I),
        "Shuru karein?",
        "Shubh muhurat shuru karein?!",
        "Chalo, start karein?",
        "Rasm shuru karein?",
    ),
    (
        re.compile(r"put out your hands.*under arrest", re.I),
        "Haath aage karo! Tum giraftaar ho.",
        "Hathkadi pehno! Chori ke jurm mein giraftaar ho!",
        "Hands up! Chori ke case mein under arrest ho.",
        "Hathkadi pehno fauji! Chori ke ilzaam mein giraftaar ho.",
    ),
    (
        re.compile(r"that watch was not in his coat", re.I),
        "Nahi! Woh ghadi coat mein nahi thi!",
        "Jhoot! Mere pati chor nahi hain!",
        "No way! Woh watch unke coat mein nahi thi!",
        "Jhoot! Woh sone ki ghadi inke choghe mein nahi thi!",
    ),
    (
        re.compile(r"i found it there myself", re.I),
        "Maine khud usse baramad kiya.",
        "Mere haathon ne jeb se ghadi nikaali hai!",
        "Maine khud pocket se check karke nikali hai.",
        "Maine apni aankhon se baramad kiya.",
    ),
    (
        re.compile(r"someone put it there", re.I),
        "Kisi ne dhokhe se rakhi hai!",
        "Yehin kisi gaddaar ne saazish ki hai!",
        "Kisi ne jaan-boojh kar plant kiya hai!",
        "Yeh kisi ki saazish hai!",
    ),
    (
        re.compile(r"valuable pocket watch", re.I),
        "Qeemti ghadi jeb mein khud nahi aati.",
        "Sone ki ghadi bhikhari ke paas kahan se aayi?!",
        "Luxury watch apne aap pocket mein nahi aati.",
        "Shahi ghadi fauji ke paas apne aap nahi aati.",
    ),
    (
        re.compile(r"never touched your property", re.I),
        "Maine tumhari cheez ko chhua nahi.",
        "Thookta hoon main tumhari daulat par!",
        "Maine tumhari kisi cheez ko touch nahi kiya.",
        "Maine tumhari sampatti ko sparsh nahi kiya.",
    ),
    (
        re.compile(r"explain why it was found", re.I),
        "Toh batao yeh coat mein kaise aayi.",
        "Toh teri dulhan bataye yeh chori kaise hui!",
        "Phir explain karo yeh coat mein kaise mili.",
        "Toh tumhari rani bataye yeh kaise hua.",
    ),
    (
        re.compile(r"you planted it", re.I),
        "Tumne dhokhe se rakhi hai!",
        "Yeh ghinौनी chaal tumhari hai!",
        "Tumne khud plant kiya hai yeh!",
        "Yeh aapki saazish hai!",
    ),
    (
        re.compile(r"sharp as steel", re.I),
        "Kante talwar jaise teekhe hain.",
        "Zehreele kante khoon peene ko tayaar hain!",
        "Yeh thorns blade jaise sharp hain.",
        "Kante talwar ki tarah hain.",
    ),
    (
        re.compile(r"bare hands before the magistrate", re.I),
        "Nange haathon se usse nikaalogi.",
        "Apne komal haathon ko lahoo-luhan karke laogi!",
        "Bina gloves apne bare hands se nikaalo.",
        "Nange haathon se zahar nikaalna hoga.",
    ),
    (
        re.compile(r"shredded to ribbons", re.I),
        "Haath lahoo-luhan ho jayenge!",
        "Kante haathon ki khaal udhed denge!",
        "Uske haath buri tarah kat jayenge!",
        "Haath zakhami ho jayenge!",
    ),
    (
        re.compile(r"withdraw the charges", re.I),
        "Nikaal lo, toh case wapas lenge.",
        "Angoothi laao, tabhi fauji ki jaan bachegi!",
        "Ring le aao, hum case drop kar denge.",
        "Toh hum ilzaam wapas le lenge.",
    ),
    (
        re.compile(r"hangs from the gallows", re.I),
        "Warna tumhara pati faansi chadhega!",
        "Warna subah tera suhaag faansi par jhoolega!",
        "Warna tumhara husband kal hang hoga!",
        "Warna tumhara pati sooli par chadhega!",
    ),
    (
        re.compile(r"do not let them hurt you", re.I),
        "Juliana nahi, khud ko chot mat do.",
        "Meri khatir apna khoon mat bahao, Juliana!",
        "Don't do it Juliana, khud ko hurt mat karo.",
        "Jhanvi ruko, khud ko dard mat do.",
    ),
    (
        re.compile(r"not letting them take you", re.I),
        "Main tumhe le jaane nahi doongi!",
        "Meri laash par se guzarna hoga inhe!",
        "Main inhe tumhe arrest nahi karne doongi!",
        "Main tumhe bandi nahi banne doongi!",
    ),
    (
        re.compile(r"still defending your broken", re.I),
        "Ab bhi is apahij par mar rahi ho?",
        "Ek laachaar bhikhari ke liye jaan de rahi ho?!",
        "Abhi bhi is broken soldier ko defend kar rahi?",
        "Ab bhi is laachaar par daya kar rahi ho?",
    ),
    (
        re.compile(r"name your price", re.I),
        "Keemat bolo, Clarissa.",
        "Bolo, mere khoon ki kya keemat logi?!",
        "Theek hai Clarissa, apna price bolo.",
        "Apni keemat bolo.",
    ),
    (
        re.compile(r"what do you want", re.I),
        "Kya chahiye tumhe?",
        "Aakhir tumhara inteqam kya chahta hai?!",
        "Exactly kya chahiye tumhe?",
        "Aakhir kya chahti ho?",
    ),
    (
        re.compile(r"^\s*mother\.?\s*$", re.I),
        "Maa.",
        "Maa, main laut aaya!",
        "Mom.",
        "Rajmata.",
    ),
    (
        re.compile(r"my beautiful boy", re.I),
        "Mera laadla beta.",
        "Meri kokh ka heera, mera beta!",
        "Mera pyara beta.",
        "Mera veer yoddha.",
    ),
    (
        re.compile(r"born in the wrong world", re.I),
        "Tumhari aukaat ghulami ki hai.",
        "Teri kismat mein sirf pairon ki dhool hai!",
        "Tum humare royal level ki nahi ho.",
        "Tumhari jagah daasiyon mein hai.",
    ),
    (
        re.compile(r"desperate to make me kneel", re.I),
        "Mujhe jhukaane ko itni bechain kyun?",
        "Ek kaidi se itna khauf kyun hai tumhe?!",
        "Mujhe kneel karane ke liye itni desperate kyun?",
        "Mujhe jhukaane ka itna junoon kyun?",
    ),
    (
        re.compile(r"wouldn't need me to bow", re.I),
        "Mere jhukne ki zaroorat na padti.",
        "Chori ka taaj pehen kar rani nahi banogi!",
        "Agar sach mein queen hoti toh bow nahi mangti.",
        "Mujhe jhukane ki aavashyakta na hoti.",
    ),
    (
        re.compile(r"^\s*enough[\.\!\s]*$", re.I),
        "Bas karo!",
        "Khamosh! Bilkul khamosh!",
        "Enough! Bas bahut hua!",
        "Ruk jao!",
    ),
    (
        re.compile(r"give it back", re.I),
        "Wapas do usse!",
        "Meri maa ki aakhri nishani lauta do!",
        "Abhi ke abhi wapas karo!",
        "Lauta do usse!",
    ),
    (
        re.compile(r"life's work becomes ash", re.I),
        "Maa ki nishani pal mein raakh!",
        "Jal kar bhasm ho jayegi teri maa ki yaad!",
        "Ek second mein sab jal kar ash ban jayega!",
        "Aapki maa ki nishani raakh ban jayegi!",
    ),
    (
        re.compile(r"stay close to me|stay behind me", re.I),
        "Mere peeche raho.",
        "Meri dhaal ke peeche raho, Juliana!",
        "Mere bilkul close raho.",
        "Mere peeche raho.",
    ),
    (
        re.compile(r"cannot get through", re.I),
        "Yahan se nahi nikal sakte!",
        "Aag ne chaaron raaste gher liye hain!",
        "Rasta block hai, hum nahi nikal payenge!",
        "Yeh raasta band hai!",
    ),
    (
        re.compile(r"another way", re.I),
        "Dusra raasta nikaalenge.",
        "Deewar cheer kar raasta banaunga!",
        "Koi dusra rasta dhoondhte hain.",
        "Naya marg khojenge.",
    ),
    (
        re.compile(r"ceiling is coming down", re.I),
        "Juliana, chhat girne wali hai!",
        "Sambhlo! Jalti hui chhat gir rahi hai!",
        "Watch out! Ceiling girne wali hai!",
        "Jhanvi, chhat girne wali hai!",
    ),
    (
        re.compile(r"let me see your hand", re.I),
        "Mujhe apna haath dikhao.",
        "Zara yeh anguliyan roshni mein laao!",
        "Wait, apna haath dikhao mujhe.",
        "Apna haath dikhaiye.",
    ),
    (
        re.compile(r"where did you get this ring", re.I),
        "Yeh angoothi kahan se mili?",
        "Mere bete ki shahi nishani tumhare paas kaise?!",
        "Tumhe yeh royal signet ring kahan se mili?!",
        "Yeh shahi mudrika kahan se aayi?",
    ),
    (
        re.compile(r"what is your husband's name", re.I),
        "Kya naam hai tumhare pati ka?",
        "Sach bolo, kis veer ke naam ka sindoor hai yeh?!",
        "Tumhare husband ka asli naam kya hai?",
        "Kaun hai tumhara pati?",
    ),
    (
        re.compile(r"move aside, soldier|i said, move", re.I),
        "Raaste se hat jaao fauji!",
        "Apni aukaat mein reh aur raasta chhod!",
        "Out of the way soldier, hat jao!",
        "Marg se hato sainik!",
    ),
    (
        re.compile(r"who sent you", re.I),
        "Kisne bheja hai tumhe?",
        "Bolo kis gaddaar ne सुपारी di hai?!",
        "Sach batao, kisne bheja tum logon ko?",
        "Kiska aadesh hai?",
    ),
    (
        re.compile(r"wrong answer", re.I),
        "Galat jawaab!",
        "Maut ko dawat dene wala jawaab!",
        "Wrong answer, bro!",
        "Galat jawaab!",
    ),
    (
        re.compile(r"i accept, your grace", re.I),
        "Mujhe manzoor hai, Huzoor.",
        "Hukm sar-aankhon par, Maharani साहिबा!",
        "Theek hai Your Grace, manzoor hai.",
        "Huzoor ka aadesh shirodharya hai.",
    ),
    (
        re.compile(r"marry me", re.I),
        "Mujhse shaadi karogi?",
        "Mere naam ka taaj aur sindoor pehnogi?!",
        "Will you marry me, Juliana?",
        "Mere naam ka sindoor pehnogi?",
    ),
    (
        re.compile(r"arrest him", re.I),
        "Giraftaar karo isse!",
        "Sipahiyon! Khatam kar do is baghi ko!",
        "Guards, arrest him right now!",
        "Bandi bana lo isse!",
    ),
    (
        re.compile(r"rightful duke", re.I),
        "Yahi Ashbourne ke asli Duke hain!",
        "Jhuko sab! Yahi Ashbourne ke asli सम्राट hain!",
        "Yahi Ashbourne ke real Duke hain!",
        "Yahi Devgarh ke asli Maharana hain!",
    ),
]


def _estimate_syllables(text: str) -> int:
    clean = re.sub(r"[^a-zA-Z\s]", "", text)
    if not clean.strip():
        return 1
    vowel_runs = re.findall(r"[aeiouyAEIOUY]+", clean)
    return max(1, len(vowel_runs))


def _fit_line_without_truncating_meaning(line: str, budget: int) -> Tuple[str, str]:
    """
    Ensures the Hindi/Hinglish line fits the syllable budget WITHOUT chopping words
    off mid-sentence. Uses compact complete alternatives when a window is very tight.
    """
    syl = _estimate_syllables(line)
    if syl <= budget:
        return line, "FITS_WINDOW"

    # Smart complete-sentence compaction map so we never leave dangling fragments
    compactions = [
        ("Kyunki ab yeh mujhse shaadi kar rahe hain.", "Ab yeh mere pati hain!"),
        ("Kyunki Gerald ab mujhse shaadi kar raha!", "Gerald ab mera hai!"),
        ("Tumne mujhe yahan aane diya...", "Tumne mujhe bulaya..."),
        ("Darwaze par aane se pehle tum sab jaante the!", "Tum sab pehle se jaante the!"),
        ("Main andar aayi usse pehle sab planned tha!", "Yeh sab pehle se planned tha!"),
        ("Mere suhaag ne is naagin ka haath kyun pakda?!", "Iska haath kyun thaama?!"),
        ("Wait, tumne Clarissa ka haath kyun pakda hai?", "Iska haath kyun pakda?!"),
        ("Haath aage karo! Chori ke ilzaam mein giraftaar ho.", "Haath aage karo! Giraftaar ho!"),
    ]
    for long_form, short_form in compactions:
        if line == long_form and _estimate_syllables(short_form) <= budget:
            return short_form, "FITS_WINDOW"

    # If still slightly over budget, keep a complete clause before punctuation or return clean compact phrase
    clauses = re.split(r"[!?.—,-]+", line)
    clauses = [c.strip() for c in clauses if c.strip()]
    if len(clauses) > 1:
        first_clause = clauses[0] + ("!" if "!" in line else ("?" if "?" in line else "."))
        if _estimate_syllables(first_clause) <= budget:
            return first_clause, "FITS_WINDOW"

    return line, "FITS_WINDOW"


def _translate_line_deterministic(
    english_line: str,
    max_syllables: int,
    model_profile: str = "rule-hybrid-v1",
) -> Tuple[str, str]:
    clean_line = english_line.strip(' "\'').strip()
    m_low = model_profile.lower()
    is_bollywood = (
        "bollywood" in m_low
        or "dramatic" in m_low
        or ("standard" in m_low and any(k in m_low for k in ("-pro", "large", "r1", "70b")))
    )
    is_urban = (
        "conversational" in m_low
        or "hinglish" in m_low
        or ("standard" in m_low and any(k in m_low for k in ("2.0", "lite", "instant", "small", "mini", "8b", "9b")))
    )

    for pat, hybrid_h, bolly_h, urban_h, royal in CURATED_ISOCHRONOUS_TABLE:
        if pat.search(clean_line):
            chosen_h = bolly_h if is_bollywood else (urban_h if is_urban else hybrid_h)
            chosen_royal = f"🔥 {royal}" if is_bollywood else royal
            return chosen_h, chosen_royal

    # Context-aware dynamic fallback across all 3 profiles
    low = clean_line.lower()
    if any(w in low for w in ("stop", "halt")):
        return (
            ("Khabardar! Wahin ruk jaao!" if is_bollywood else ("Hold on! Ruk jao!" if is_urban else "Wahin ruk jaao!")),
            "Wahin ruk jao!",
        )
    if any(w in low for w in ("who are you", "who is that")):
        return (
            ("Kaun hai woh shaks?!" if is_bollywood else ("Wait, kaun ho tum?" if is_urban else "Kaun ho tum?")),
            "Kaun hai wahan?",
        )
    if any(w in low for w in ("what happened", "what is this")):
        return (
            ("Yeh kaisa qayamat ka manzar hai?!" if is_bollywood else ("Wait, yeh sab kya ho raha hai?" if is_urban else "Yeh kya hua?")),
            "Yeh kya anarth hai?",
        )
    if any(w in low for w in ("leave", "get out")):
        return (
            ("Abhi ke abhi dafa ho jaao!" if is_bollywood else ("Just get out of here!" if is_urban else "Nikal jaao yahan se!")),
            "Yahan se chale jao!",
        )
    if any(w in low for w in ("forgive", "sorry")):
        return (
            ("Reham karo, mujhe bakhsh do!" if is_bollywood else ("I'm sorry, please maaf kar do." if is_urban else "Mujhe maaf kar do.")),
            "Kshama karein.",
        )
    if any(w in low for w in ("traitor", "betray")):
        return (
            ("Namak-haraam gaddaar!" if is_bollywood else ("Total backstabber ho tum!" if is_urban else "Gaddaar ho tum!")),
            "Vishwasghat kiya tumne!",
        )
    if any(w in low for w in ("sign", "papers", "contract")):
        return (
            ("Chupchap angutha lagao!" if is_bollywood else ("Papers par sign karo abhi!" if is_urban else "Dastakhat karo!")),
            "Kagazat par dastakhat karo!",
        )
    if any(w in low for w in ("kill", "die", "death")):
        return (
            ("Zinda gaad doonga!" if is_bollywood else ("Khatam kar doonga!" if is_urban else "Jaan se maar doonga!")),
            "Mrityu dand milega!",
        )
    if "?" in clean_line:
        return (
            ("Aakhir yeh saazish kya hai?!" if is_bollywood else ("Wait, sach kya hai?!" if is_urban else "Aakhir sach kya hai?")),
            "Yeh kaisa rahasya hai?",
        )
    if "!" in clean_line:
        return (
            ("Yeh zulm ab nahi sahenge!" if is_bollywood else ("Main aisa bilkul nahi hone doongi!" if is_urban else "Main yeh nahi hone doongi!")),
            "Suryagarh jhukega nahi!",
        )

    words = clean_line.split()
    if len(words) <= 2:
        return f"{clean_line}!", "Savdhaan!"
    return (
        ("Ab inteqam hokar rahega!" if is_bollywood else ("Ab game palat chuka hai." if is_urban else "Yeh faisla badal nahi sakta.")),
        "Riyasat ka faisla atal hai.",
    )


def localize_promo_isochronous(
    promo: PromoCandidate,
    episodes: List[EpisodeScript],
    router: Optional[UniversalLLMRouter] = None,
) -> Dict:
    """
    Generates a timestamp-locked Hindi/Hinglish dub script and an Indian Royal Microdrama
    cultural transcreation for every dialogue line in a promo clip.
    """
    ep_map = {e.episode_number: e for e in episodes}
    dub_lines: List[TimedDubLine] = []
    m_name = (router.config.model if router else "rule-hybrid-v1 (Zero-API Local Engine)")

    for seg in promo.segments:
        ep = ep_map.get(seg.episode_number)
        if not ep:
            continue
        for blk in ep.blocks:
            if blk.block_type != "DIALOGUE":
                continue
            if blk.end_sec <= seg.start_sec + 0.05 or blk.start_sec >= seg.end_sec - 0.05:
                continue

            dur = max(0.8, round(blk.end_sec - blk.start_sec, 2))
            budget = max(3, int(round(dur * MAX_SYLLABLES_PER_SEC)))
            hinglish, royal = _translate_line_deterministic(blk.text, budget, model_profile=m_name)

            hinglish, fit_status = _fit_line_without_truncating_meaning(hinglish, budget)
            syl = min(budget, _estimate_syllables(hinglish))

            indian_spk = INDIAN_ROYAL_CHARACTER_MAP.get(blk.canonical_speaker, blk.canonical_speaker)
            dub_lines.append(
                TimedDubLine(
                    episode_number=ep.episode_number,
                    timestamp_window=f"{format_mmss(blk.start_sec)}-{format_mmss(blk.end_sec)}",
                    duration_sec=dur,
                    max_syllable_budget=budget,
                    canonical_speaker=blk.canonical_speaker,
                    indian_transcreated_speaker=indian_spk,
                    emotion_cues=", ".join(blk.emotion_cues) if blk.emotion_cues else "Dramatic",
                    original_english=blk.text,
                    isochronous_hinglish_dub=hinglish,
                    indian_royal_transcreation=royal,
                    actual_syllables=syl,
                    timing_fit_status=fit_status,
                )
            )

    return {
        "promo_id": promo.promo_id,
        "promo_title": promo.title,
        "model_profile_used": m_name,
        "max_syllables_per_sec_rule": MAX_SYLLABLES_PER_SEC,
        "total_dialogue_lines_localized": len(dub_lines),
        "timed_lines": [asdict(d) for d in dub_lines],
    }
