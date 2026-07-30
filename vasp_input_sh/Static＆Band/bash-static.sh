#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=52
#SBATCH --output=%j_band.out
#SBATCH --error=%j_band.err
#SBATCH --partition=your partion

# --- 配置区域 ---
# 任务超时时间 (静态+能带计算总和)
TASK_TIMEOUT="4h" 
# 磁盘空间监控阈值 (单位: GB), 2TB = 2048 GB
DISK_LIMIT_GB=2048
# 监控的挂载点
DISK_MOUNT_POINT="/share"
# ----------------

ulimit -s unlimited
module load your vasp


POSCAR_PATTERN="*.vasp"
declare -a POSCAR_FILES=()

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}

# 输入目录指向 successCONTCAR
POSCAR_DIR="${BASE_DIR}/successCONTCAR"

# 定义磁矩字典
declare -A MAG_VALUES=(
    ["Sc"]=1.0 ["Ti"]=2.0 ["V"]=3.0 ["Cr"]=5.0 ["Mn"]=5.0
    ["Fe"]=4.0 ["Co"]=3.0 ["Ni"]=2.0 ["Cu"]=1.0 ["Zn"]=0
    ["Y"]=1.0 ["Zr"]=2.0 ["Nb"]=5.0 ["Mo"]=5.0 ["Tc"]=5.0
    ["Ru"]=4.0 ["Rh"]=3.0 ["Pd"]=0 ["Ag"]=1.0 ["Cd"]=0
    ["Hf"]=2.0 ["Ta"]=3.0 ["W"]=4.0 ["Re"]=5.0 ["Os"]=4.0
    ["Ir"]=3.0 ["Pt"]=2.0 ["Au"]=1.0 ["Hg"]=0
    ["La"]=0 ["Ce"]=1.0 ["Pr"]=2.0 ["Nd"]=3.0 ["Pm"]=4.0
    ["Sm"]=5.0 ["Eu"]=7.0 ["Gd"]=7.0 ["Tb"]=6.0 ["Dy"]=5.0
    ["Ho"]=4.0 ["Er"]=3.0 ["Tm"]=2.0 ["Yb"]=1.0 ["Lu"]=0
    ["Ac"]=0 ["Th"]=0 ["Pa"]=2.0 ["U"]=3.0 ["Np"]=4.0
    ["Pu"]=5.0 ["Am"]=6.0 ["Cm"]=7.0 ["Bk"]=6.0 ["Cf"]=5.0
    ["Es"]=4.0 ["Fm"]=3.0 ["Md"]=2.0 ["No"]=1.0 ["Lr"]=0
    ["H"]=0 ["He"]=0 ["Li"]=0 ["Be"]=0 ["B"]=0 ["C"]=0 ["N"]=0 ["O"]=0 ["F"]=0 ["Ne"]=0
    ["Na"]=0 ["Mg"]=0 ["Al"]=0 ["Si"]=0 ["P"]=0 ["S"]=0 ["Cl"]=0 ["Ar"]=0
    ["K"]=0 ["Ca"]=0 ["Ga"]=0 ["Ge"]=0 ["As"]=0 ["Se"]=0 ["Br"]=0 ["Kr"]=0
    ["Rb"]=0 ["Sr"]=0 ["In"]=0 ["Sn"]=0 ["Sb"]=0 ["Te"]=0 ["I"]=0 ["Xe"]=0
    ["Cs"]=0 ["Ba"]=0 ["Tl"]=0 ["Pb"]=0 ["Bi"]=0 ["Po"]=0 ["At"]=0 ["Rn"]=0
    ["Fr"]=0 ["Ra"]=0
)

load_poscar_files() {
    shopt -s nullglob
    POSCAR_FILES=(${POSCAR_DIR}/${POSCAR_PATTERN})
    shopt -u nullglob
    if [ ${#POSCAR_FILES[@]} -eq 0 ]; then
        return
    fi
    local old_ifs="${IFS}"
    IFS=$'\n' POSCAR_FILES=($(printf '%s\n' "${POSCAR_FILES[@]}" | sort))
    IFS="${old_ifs}"
}

# --- 检查磁盘空间函数 ---
check_disk_space() {
    local avail_gb=$(df -BG "${DISK_MOUNT_POINT}" | tail -1 | awk '{print $4}' | sed 's/G//')
    if [[ -z "${avail_gb}" ]]; then
        echo "WARNING: 无法获取磁盘空间信息，跳过检查。"
        return 0
    fi
    echo ">>> 磁盘检查: ${DISK_MOUNT_POINT} 剩余: ${avail_gb} GB (阈值: ${DISK_LIMIT_GB} GB)"
    if [ "${avail_gb}" -lt "${DISK_LIMIT_GB}" ]; then
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "CRITICAL: 磁盘空间不足 2TB！剩余: ${avail_gb} GB"
        echo "停止所有计算任务。"
        echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        echo "$(date '+%Y-%m-%d %H:%M:%S') - 磁盘不足停止" >> "${BASE_DIR}/DISK_FULL_STOP.log"
        exit 99
    fi
}

if [ "$1" = "run" ]; then
    shift
    if [ $# -lt 2 ]; then
        echo "运行模式错误"
        exit 1
    fi
    BATCH_START=$1
    BATCH_END=$2
    load_poscar_files

    ROOT_DIR=$(pwd)
    STATIC_DIR="${ROOT_DIR}/static_calc"
    mkdir -p "${STATIC_DIR}"
    
    RESULTS_LOG="${ROOT_DIR}/workflow_results.txt"
    FAIL_LOG="${ROOT_DIR}/workflow_failed_jobs.txt"

    VASP_PROCS=${SLURM_NTASKS:-${SLURM_NPROCS:-40}}

    # --- 核心处理函数 ---
    process_structure() {
        local task="$1"
        set +e

        # 1. 每次任务开始前检查磁盘
        check_disk_space

        CURRENT_POSCAR=${POSCAR_FILES[$((task - 1))]}
        STRUCT_LABEL=$(basename "${CURRENT_POSCAR}" .vasp)
        
        echo "=== 处理任务 ${task}: ${STRUCT_LABEL} ==="

        if [ ! -f "${CURRENT_POSCAR}" ]; then
            echo "未找到 POSCAR: ${STRUCT_LABEL}" >> "${FAIL_LOG}"
            return 0
        fi

        # ==========================================
        # 阶段 1: 静态自洽计算 (Static SCF)
        # ==========================================
        WORK_DIR="${STATIC_DIR}/calc_${STRUCT_LABEL}"
        mkdir -p "${WORK_DIR}"
        cd "${WORK_DIR}" || return 0

        # 准备文件
        cp "${CURRENT_POSCAR}" POSCAR
        
        # 生成 POTCAR & KPOINTS (Grid)
        # 静态计算使用 K-Spacing 0.03 (Vaspkit 102)
        echo -e "102\n2\n0.03" | vaspkit > /dev/null 2>&1

        # 解析 MAGMOM
        ELEMENTS=$(sed -n '6p' POSCAR | tr -d '\r')
        ATOM_NUMS=$(sed -n '7p' POSCAR | tr -d '\r')
        ELEM_ARRAY=()
        for elem in ${ELEMENTS}; do ELEM_ARRAY+=("${elem}"); done
        NUM_ARRAY=()
        for count in ${ATOM_NUMS}; do NUM_ARRAY+=("${count}"); done

        MAGMOM_VALUES=()
        for i in "${!ELEM_ARRAY[@]}"; do
            elem="${ELEM_ARRAY[$i]}"
            num="${NUM_ARRAY[$i]}"
            raw_mag="${MAG_VALUES[$elem]:-0}"
            [[ "${raw_mag}" == *.* ]] && mag="${raw_mag}" || mag="${raw_mag}.0"
            for ((j=0; j<num; j++)); do MAGMOM_VALUES+=("${mag}"); done
        done
        MAGMOM_STR=$(IFS=' '; printf '%s' "${MAGMOM_VALUES[*]}")

        # 处理 LMAXMIX (f 电子)
        LMAXMIX_VAL=4
        if [[ "${ELEMENTS}" =~ "Eu" ]] || [[ "${ELEMENTS}" =~ "Gd" ]] || [[ "${ELEMENTS}" =~ "Ce" ]]; then
            LMAXMIX_VAL=6
        fi

        # 生成 INCAR (Static)
        # 这一步必须计算电荷 (ICHARG=2, LCHARG=T) 供下一步使用
        cat > INCAR <<EOF
SYSTEM = Static Calculation
ISTART = 0
ICHARG = 2
LREAL = .FALSE.  ! 单点能计算建议关闭 LREAL 以获得高精度能量 (大体系可改为 Auto)

ENCUT = 500
PREC = Accurate
EDIFF = 1E-5
ISMEAR = 1       ! 保持与Relax一致，或改为 -5 (Tetrahedron) 用于态密度
SIGMA = 0.2
ALGO = Normal
MAGMOM = ${MAGMOM_STR}
NELM = 300
LMAXMIX = 4   

ISPIN = 2
AMIX = 0.2
BMIX = 0.0001
AMIX_MAG = 0.8
BMIX_MAG = 0.0001


IBRION = -1      ! 离子不移动
NSW = 0          ! 步数为0
ISIF = 2         ! 计算力和应力，但不改变晶胞
POTIM = 0.2

LORBIT = 11      ! 输出态密度 (DOS)
LVHAR = .TRUE.
LELF  = .TRUE.
LWAVE = .TRUE.
LCHARG = .TRUE.
NBANDS = 52
NPAR = 4
IVDW = 11

EOF

        echo ">>> [1/2] 启动 Static Calculation..."
        timeout --kill-after=5m ${TASK_TIMEOUT} mpirun -np ${VASP_PROCS} vasp_std
        STAT_EXIT=$?

        # 检查 Static 是否成功
        SUCCESS_STATIC=0
        if [ $STAT_EXIT -eq 0 ] && (grep -q "Voluntary" OUTCAR || grep -q "General timing" OUTCAR); then
            SUCCESS_STATIC=1
            E_TOTAL=$(grep "energy  without" OUTCAR | tail -1 | awk '{print $4}')
            echo "Static 完成: Energy = ${E_TOTAL} eV"
        else
            echo "Static 失败或超时，跳过 Band 计算。" >> "${FAIL_LOG}"
            cd "${ROOT_DIR}"
            return 0
        fi

        # ==========================================
        # 阶段 2: 能带计算 (Band Non-SCF)
        # ==========================================
        BAND_DIR="${WORK_DIR}/Band"
        mkdir -p "${BAND_DIR}"
        cd "${BAND_DIR}" || return 0

        # 准备文件
        cp ../POSCAR .
        cp ../POTCAR .
        cp ../CHGCAR .  # 关键：读取电荷密度
        
        # 生成固定格式的 KPOINTS (M-K-G-M)
        cat > KPOINTS <<EOF
K-Path M-K-G-M high density
150
Line-mode
Reciprocal
0.5000 0.0000 0.0000 ! M
0.3333 0.3333 0.0000 ! K

0.3333 0.3333 0.0000 ! K
0.0000 0.0000 0.0000 ! G

0.0000 0.0000 0.0000 ! G
0.5000 0.0000 0.0000 ! M
EOF

        # 生成 INCAR (Band) - 使用您提供的特定参数
        cat > INCAR <<EOF
SYSTEM = Static Calculation
ISTART = 0
ICHARG = 11
LREAL = .FALSE.  ! 单点能计算建议关闭 LREAL 以获得高精度能量 (大体系可改为 Auto)

ENCUT = 500
PREC = Accurate
EDIFF = 1E-5
ISMEAR = 1       ! 保持与Relax一致，或改为 -5 (Tetrahedron) 用于态密度
SIGMA = 0.2
ALGO = Normal
MAGMOM = ${MAGMOM_STR}
NELM = 300
LMAXMIX = 4   

ISPIN = 2
AMIX = 0.2
BMIX = 0.0001
AMIX_MAG = 0.8
BMIX_MAG = 0.0001


IBRION = -1      ! 离子不移动
NSW = 0          ! 步数为0
ISIF = 2         ! 计算力和应力，但不改变晶胞

LORBIT = 11      ! 输出态密度 (DOS)
LVHAR = .TRUE.
LELF  = .TRUE.
LWAVE = .FALSE.
LCHARG = .FALSE.
NBANDS = 52
NPAR = 4
IVDW = 11

EOF

        echo ">>> [2/2] 启动 Band Calculation..."
        timeout --kill-after=5m ${TASK_TIMEOUT} mpirun -np ${VASP_PROCS} vasp_std
        BAND_EXIT=$?

        if [ $BAND_EXIT -eq 0 ]; then
            echo "${STRUCT_LABEL}: Static & Band 完成" >> "${RESULTS_LOG}"
        else
            echo "${STRUCT_LABEL}: Static 成功, Band 失败" >> "${FAIL_LOG}"
        fi

        cd "${ROOT_DIR}"
        return 0
    }

    # 循环执行
    for ((task=${BATCH_START}; task<=${BATCH_END}; task++)); do
        process_structure "${task}"
    done
    exit 0
fi

# --- 提交部分 ---
if [ $# -ne 0 ]; then
    echo "使用方法: $0"
    exit 1
fi

load_poscar_files
if [ ${#POSCAR_FILES[@]} -eq 0 ]; then
    echo "未在 ${POSCAR_DIR} 中找到文件。"
    exit 1
fi

TOTAL_TASKS=${#POSCAR_FILES[@]}
BATCH_COUNT=4
BATCH_SIZE=$(( (TOTAL_TASKS + BATCH_COUNT - 1) / BATCH_COUNT ))

echo "共 ${TOTAL_TASKS} 个任务，来自 ${POSCAR_DIR}"
echo "开启硬盘监控 (阈值: ${DISK_LIMIT_GB} GB)"

for ((batch=0; batch<BATCH_COUNT; batch++)); do
    BATCH_START=$(( batch * BATCH_SIZE + 1 ))
    if [ ${BATCH_START} -gt ${TOTAL_TASKS} ]; then break; fi
    BATCH_END=$(( BATCH_START + BATCH_SIZE - 1 ))
    if [ ${BATCH_END} -gt ${TOTAL_TASKS} ]; then BATCH_END=${TOTAL_TASKS}; fi

    echo "提交批次 $((batch + 1)): ${BATCH_START}-${BATCH_END}"
    sbatch "$0" run ${BATCH_START} ${BATCH_END}
    sleep 1
done