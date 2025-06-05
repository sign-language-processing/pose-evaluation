import shutil
import json
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import typer

app = typer.Typer()


@app.command()
def filter_parquet_files(
    in_path: Path = typer.Argument(..., help="Folder with input .parquet files"),
    out_path: Path = typer.Argument(..., help="Folder to save filtered .parquet files"),
    min_hyp: int = typer.Option(5, help="Minimum number of unique GLOSS_A_PATH values"),
    max_hyp: int = typer.Option(20, help="Maximum number of unique GLOSS_A_PATH values"),
    glosses_to_include: str = typer.Option(None, help="Comma-separated list of glosses to include"),
    report_path: Path = typer.Option(None, help="Optional path to save a JSON report"),
):
    """Filters .parquet files based on unique GLOSS_A_PATH count and copies them to a new folder."""
    glosses_set = set(g.strip().upper() for g in glosses_to_include.split(","))
    typer.echo(f"📂 Input path: {in_path}")
    typer.echo(f"📁 Output path: {out_path}")
    typer.echo(f"Glosses to Include: {len(glosses_set)} glosses")
    typer.echo(f"🔢 min_hyp: {min_hyp}, max_hyp: {max_hyp}")
    if report_path:
        typer.echo(f"📝 Report will be saved to: {report_path}")

    out_path.mkdir(parents=True, exist_ok=True)
    parquet_files = sorted(in_path.glob("*.parquet"))
    already_existing = list(out_path.glob("*parquet"))
    typer.echo(f"🔍 Found {len(parquet_files)} .parquet files in source, {len(already_existing)} in target")

    copied_files = []

    def process_file(file: Path) -> bool:
        file_gloss = file.stem.split("_")[0]
        destination = out_path / file.name
        if file_gloss in glosses_set:
            if not destination.exists():
                shutil.copy(file, destination)
                return True
        return False

    with ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, parquet_files), total=len(parquet_files), desc="Copying files"))

    kept = sum(results)
    skipped = len(results) - kept
    typer.echo(f"✅ Copied: {kept}, Already Existed: {len(already_existing)}, ⏭️ Skipped: {skipped}")

    if report_path:
        report = {
            "already_existing": len(already_existing),
            "copied": kept,
            "skipped": skipped,
            "glosses_used": sorted(glosses_set),
            "timestamp": datetime.utcnow().isoformat(),
        }
        with report_path.open("w") as f:
            json.dump(report, f, indent=2)


if __name__ == "__main__":
    app()


# GLOSS	SAMPLES
# DECIDE2	8
# CHEW1	10
# DEAF2	10
# PIPE2	10
# CASTLE2	10
# SHARK2	11
# PJS	11
# TURBAN	11
# WASHDISHES	11
# DART	11
# FORK4	11
# DOLPHIN2	11
# CELERY	11
# CANDY1	11
# OHISEE	11
# TIE1	11
# DIP3	11
# WIFE	11
# ACCENT	11
# SHAVE5	11
# BRAG	11
# RUSSIA	11
# DRINK2	11
# DEER	12
# LEAF1	12
# SIX	12
# HARDOFHEARING	12
# FISHING2	12
# LIVE2	12
# WEIGH	12
# PERCENT	12
# SEVEN	12
# SENATE	12
# MICROPHONE	12
# TRUE	12
# ADVERTISE	12
# NEWSTOME	12
# OPINION1	12
# BAG2	12
# THANKYOU	12
# HIGHSCHOOL	12
# BANANA2	12
# DONTCARE	12
# PHILADELPHIA	12
# CALIFORNIA	12
# DONTKNOW	12
# DEAFSCHOOL	12
# PREFER	12
# COUNSELOR	12
# HUG	12
# NECKLACE4	12
# BINOCULARS	12
# NINE	12
# EIGHT	12
# RAT	12
# FRIDAY	12
# SNOWSUIT	12
# LEND	12
# BRAINWASH	12
# SCREWDRIVER3	12
# VAMPIRE	12
# ALASKA	12
# KNITTING3	12
# CONVINCE2	12
# ERASE1	12
# BERRY	12
# SHAME	12
# WEDNESDAY	13
# DONTMIND	13
# SALAD	13
# TOSS	13
# WORKSHOP	13
# NEWYORK	13
# PERFUME1	13
# NEWSPAPER	13
# BEAR	13
# CHILD	13
# MAIL1	13
# ADULT	13
# BEAK	13
# CHICAGO	13
# TUESDAY	13
# VOTE	13
# ROSE	13
# CAFETERIA	13
# COMMITTEE	13
# WEIGHT	13
# REPLACE	13
# BEER	13
# SATURDAY	13
# GALLAUDET	13
# THURSDAY	13
# SPECIALIST	13
# VOMIT	13
# GOTHROUGH	13
# ARIZONA	13
# PHONE	13
# SPICY	14
# FINGERSPELL	14
# TRADITION	14
# PEPSI	14
# DRIP	14
# HUSBAND	14
# EACH	14
# BUT	14
# SOCKS	14
# SEVERAL	14
# EASTER	14
# STADIUM	14
# EVERYTHING	14
# SORRY	14
# SNAKE	14
# LIBRARY	14
# BUFFALO	14
# MOOSE	14
# SANTA	14
# RUIN	14
# INTRODUCE	14
# ROOF	15
# CLOSE	15
# HERE	15
# WEAR	15
# TEA	15
# PRACTICE	15
# HAWAII	15
# WEATHER	15
# GOVERNMENT	15
# ROCKINGCHAIR1	15
# EXPERIMENT	15
# STARS	15
# INVITE	15
# THREE	16
# PAIR	16
# PRINT	16
# GLASSES	16
# TAKEOFF1	16
# SILVER	16
# YOUR	16
# RESTAURANT	16
# ONION	16
# PRESIDENT	17
# LUCKY	17
# ENOUGH	17
# FRIENDLY	17
# THANKSGIVING	17
# DUTY	17
# BATHROOM	17
# TOP	17
# PILL	18
# MEETING	18
# PUMPKIN	18
# FEW	18
# COOKIE	18
# EARN	18
# FOCUS	18
# APPEAR	18
# DRUG	18
# THRILLED	19
# WEST	19
# VACATION	19
# RECORDING	19
# TASTE	19
# MOTHER	19
# PIE	19
# CHALLENGE	19
# DISRUPT	19
# WINE	19
# CALM	20
# TELL	20
# FULL	21
# GIVE	21
# FAVORITE	21
# HURRY	21
# ORGANIZATION	21
# DAY	21
# SIT	22
# TODAY	22
# GOLD	22
# MY	23
# SCIENCE	24
# CHILDREN	24
# NAME	25
# RACE	25
# CLASS	26
# STUDENT	26
# KNOW	26
# HOLIDAY	26
# WINTER	28
# BAD	28
# GROUP	28
# WATCH	28
# HISTORY	29
# SHOES	29
# OPEN	29
# EXCITED	30
# MACHINE	30
# GAME	32
# WRITE	32
# SEE	33
# SUMMER	35
# ATTENTION	38
# TEAM	39
# GOOD	40
# HOUSE	40
# PAPER	43
# LOVE	44
# DIFFERENT	44
# LEARN	48
# SCHOOL	51
# SAD	51
# COLOR	51
# COLD	54
# FAMILY	63
# BIG	66
# YESTERDAY	67
# WORK	69
# ENJOY	79
# CHAIR	94
# WATER	100
# WHY	103
# PENCIL	117
# FACE	134
# BLACK	136
# AUNT	148
# COW	162
# UNCLE	166
# ICECREAM	166
# GIRL	167
# THIRSTY	167
# AIRPLANE	167
# LION	168
# HAVE	168
# APPLE	169
# HIDE	170
# TIGER	171
# MOUSE	171
# BROWN	171
# DANCE	172
# TOMORROW	173
# HORSE	173
# GRASS	173
# PLEASE	174
# PIG	175
# HUNGRY	177
# YELLOW	185
# GIFT	185
# READ	187
# PRETTY	195
# ANIMAL	213
# DIRTY	213


# cd /data/petabyte/cleong/projects/pose-eval/metric_results_round_4 && conda activate /opt/home/cleong/envs/pose_eval_src && python /opt/home/cleong/projects/pose-evaluation/pose_evaluation/evaluation/filter_parquets.py scores 169_glosses/ --glosses-to-include "SORRY,MOTHER,BEER,CALIFORNIA,DEAFSCHOOL,GOVERNMENT,FRIDAY,CHEW1,WEDNESDAY,REPLACE,THRILLED,MEETING,YOUR,SEVERAL,HAWAII,DRUG,DECIDE2,SHARK2,VOTE,HARDOFHEARING,OHISEE,PERFUME1,SCREWDRIVER3,LIBRARY,FORK4,LIVE2,CALM,SHAME,CAFETERIA,BANANA2,MOOSE,MAIL1,SANTA,BEAR,THANKSGIVING,TIE1,PAIR,SPECIALIST,ARIZONA,NECKLACE4,PRINT,DRINK2,THURSDAY,SIX,CASTLE2,TOSS,WEIGH,PRACTICE,STARS,LEAF1,HUSBAND,BEAK,CHALLENGE,BINOCULARS,DOLPHIN2,VAMPIRE,PUMPKIN,BRAINWASH,COMMITTEE,TEA,TURBAN,PREFER,EASTER,HUG,BATHROOM,RUIN,SNAKE,PHILADELPHIA,CONVINCE2,DONTKNOW,EIGHT,COOKIE,TELL,DEAF2,PIPE2,SATURDAY,SEVEN,SILVER,ROOF,DRIP,DUTY,COUNSELOR,NINE,RECORDING,RAT,SALAD,EVERYTHING,SNOWSUIT,EACH,CHICAGO,BAG2,PRESIDENT,GALLAUDET,CLOSE,FEW,CELERY,EARN,PEPSI,SOCKS,MICROPHONE,LUCKY,PJS,TRUE,ROSE,GOTHROUGH,RESTAURANT,WEATHER,STADIUM,FISHING2,PERCENT,KNITTING3,EXPERIMENT,TAKEOFF1,ACCENT,OPINION1,PIE,RUSSIA,WEIGHT,DONTCARE,ROCKINGCHAIR1,CANDY1,SPICY,ENOUGH,GLASSES,TUESDAY,WIFE,WASHDISHES,NEWSTOME,WEST,APPEAR,INTRODUCE,DONTMIND,HERE,LEND,PHONE,ERASE1,THREE,ADVERTISE,BERRY,DART,WINE,PILL,FRIENDLY,DIP3,TRADITION,TOP,ADULT,TASTE,DISRUPT,VACATION,SENATE,NEWSPAPER,FOCUS,DEER,INVITE,BRAG,BUFFALO,SHAVE5,BUT,CHILD,NEWYORK,WORKSHOP,FINGERSPELL,ALASKA,ONION,VOMIT,WEAR,THANKYOU,HIGHSCHOOL"
