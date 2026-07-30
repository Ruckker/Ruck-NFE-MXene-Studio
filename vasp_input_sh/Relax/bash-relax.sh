#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --output=%j.out
#SBATCH --error=%j.err
#SBATCH --partition=your partion


ulimit -s unlimited
module load your vasp

POSCAR_PATTERN="*-*-*-*-*-*-*.vasp"
declare -a POSCAR_FILES=()

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
BASE_DIR=${SLURM_SUBMIT_DIR:-${SCRIPT_DIR}}
POSCAR_DIR="${BASE_DIR}/poscars"

# --- 配置区域 ---
# 单个任务超时时间 (4小时)
TASK_TIMEOUT="4h" 
# ----------------


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

if [ "$1" = "run" ]; then
    shift

    if [ $# -lt 2 ]; then
        echo "运行模式需要提供起始和结束索引，例如: $0 run 1 100"
        exit 1
    fi

    BATCH_START=$1
    BATCH_END=$2

    load_poscar_files

    if [ ${#POSCAR_FILES[@]} -eq 0 ]; then
        echo "未找到 ${POSCAR_DIR}/${POSCAR_PATTERN} 文件，运行模式中止。"
        exit 1
    fi

    TOTAL_TASKS=${#POSCAR_FILES[@]}
    if [ ${BATCH_END} -gt ${TOTAL_TASKS} ]; then
        echo "结束索引 (${BATCH_END}) 超出范围 (1-${TOTAL_TASKS})"
        exit 1
    fi
    
    ROOT_DIR=$(pwd)
    RELAX_DIR="${ROOT_DIR}/relax"
    mkdir -p "${RELAX_DIR}"
    
    RESULTS_LOG="${ROOT_DIR}/relax_results.txt"
    FAIL_LOG="${ROOT_DIR}/relax_failed_jobs.txt"
    TIMEOUT_LOG="${ROOT_DIR}/relax_timeout.txt"

    MPIRUN_PATH=$(command -v mpirun 2>/dev/null)
    VASP_PROCS=${SLURM_NTASKS:-${SLURM_NPROCS:-104}}
    
    echo "工作目录: ${ROOT_DIR}"
    echo "处理结构索引范围: ${BATCH_START}-${BATCH_END} / 总共 ${TOTAL_TASKS}"
    echo "使用进程数: ${VASP_PROCS} (超时限制: ${TASK_TIMEOUT})"

    # --- 核心处理函数 ---
    process_structure() {
        local task="$1"
        set +e

        CURRENT_POSCAR=${POSCAR_FILES[$((task - 1))]}
        STRUCT_LABEL=$(basename "${CURRENT_POSCAR}")
        echo "------------------------------------------"
        echo "开始处理索引 ${task}: ${STRUCT_LABEL}"
        echo "------------------------------------------"

        if [ ! -f "${CURRENT_POSCAR}" ]; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') 未找到 POSCAR: ${STRUCT_LABEL}" >> "${FAIL_LOG}"
            return 0
        fi

        WORK_DIR="${RELAX_DIR}/calc_${STRUCT_LABEL}"
        
        # --- 新增功能：智能判断是否续算 ---
        local use_existing_contcar=0
        local need_cleanup=0

        if [ -d "${WORK_DIR}" ]; then
            # 检查 OUTCAR 是否有正常结束标志
            if [ -f "${WORK_DIR}/OUTCAR" ] && grep -q "General timing" "${WORK_DIR}/OUTCAR"; then
                # 检查 CONTCAR 是否存在且不为空
                if [ -s "${WORK_DIR}/CONTCAR" ]; then
                    echo ">>> 检测到已完成的计算，将使用 CONTCAR 进行再优化(续算) <<<"
                    use_existing_contcar=1
                else
                    echo ">>> 计算虽完成但 CONTCAR 丢失/为空，准备清理重算 <<<"
                    need_cleanup=1
                fi
            else
                echo ">>> 检测到上次计算未正常完成(中断/失败)，准备清理重算 <<<"
                need_cleanup=1
            fi
        fi

        mkdir -p "${WORK_DIR}"
        if ! cd "${WORK_DIR}"; then
            echo "无法进入目录: ${WORK_DIR}" >> "${FAIL_LOG}"
            return 0
        fi

        # 根据判断执行操作
        if [ ${use_existing_contcar} -eq 1 ]; then
            # 备份旧数据 (带时间戳)
            TIMESTAMP=$(date +%Y%m%d_%H%M%S)
            BACKUP_DIR="run_ok_${TIMESTAMP}"
            mkdir -p "${BACKUP_DIR}"
            # 移动关键文件到备份，保留 CONTCAR 和 POTCAR
            mv OUTCAR vasprun.xml DOSCAR EIGENVAL CHGCAR WAVECAR OSZICAR "${BACKUP_DIR}/" 2>/dev/null
            cp CONTCAR "${BACKUP_DIR}/" # 备份一份 CONTCAR
            
            # 将 CONTCAR 变为新的 POSCAR
            cp CONTCAR POSCAR
        else
            if [ ${need_cleanup} -eq 1 ]; then
                echo "正在清理旧文件..."
                rm -f * 
                fi
            # 复制原始 POSCAR
            if ! /bin/cp -f -- "${CURRENT_POSCAR}" POSCAR; then
                echo "复制 POSCAR 失败" >> "${FAIL_LOG}"
                cd "${ROOT_DIR}"
                return 0
            fi
        fi

        # 清理当前运行产生的临时文件 (防止上次残留干扰)
        rm -f OUTCAR vasprun.xml DOSCAR EIGENVAL CHGCAR WAVECAR CONTCAR OSZICAR

        # 2. 生成 POTCAR (即使是续算，也建议重新生成以防万一)
        if ! echo -e "102\n2\n0.04" | vaspkit > /dev/null 2>&1; then
            echo "$(date '+%Y-%m-%d %H:%M:%S') vaspkit 失败: ${STRUCT_LABEL}" >> "${FAIL_LOG}"
            cd "${ROOT_DIR}"
            return 0
        fi

        # 3. 解析元素与磁矩
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
            if ! [[ "${num}" =~ ^[0-9]+$ ]]; then MAGMOM_VALUES=(); break; fi
            raw_mag="${MAG_VALUES[$elem]:-0}"
            [[ "${raw_mag}" == *.* ]] && mag="${raw_mag}" || mag="${raw_mag}.0"
            for ((j=0; j<num; j++)); do MAGMOM_VALUES+=("${mag}"); done
        done

        if [ ${#MAGMOM_VALUES[@]} -eq 0 ]; then
            echo "MAGMOM 生成错误" >> "${FAIL_LOG}"
            cd "${ROOT_DIR}"
            return 0
        fi
        MAGMOM_STR=$(IFS=' '; printf '%s' "${MAGMOM_VALUES[*]}")

        # 4. 生成 INCAR (稳健版)
        LMAXMIX_VAL=4
        if [[ "${ELEMENTS}" =~ "Eu" ]] || [[ "${ELEMENTS}" =~ "Gd" ]] || [[ "${ELEMENTS}" =~ "Ce" ]]; then
            LMAXMIX_VAL=6
        fi

        cat > INCAR <<EOF
SYSTEM = Relax
ISTART = 0
ICHARG = 2
LREAL = Auto

ENCUT = 500
PREC = Accurate
EDIFF = 1E-5
ISMEAR = 1
SIGMA = 0.2
ALGO = Normal      
NELM = 150         
LMAXMIX = ${LMAXMIX_VAL}     

ISPIN = 2
MAGMOM = ${MAGMOM_STR}
AMIX = 0.2
BMIX = 0.0001 
AMIX_MAG = 0.8
BMIX_MAG = 0.0001

IBRION = 2 
POTIM = 0.3        
NSW = 300
ISIF = 3
EDIFFG = -0.02
POTIM = 0.2        
NBANDS = 40
NPAR = 4
LWAVE = .F.
LCHARG = .F.
IVDW = 11
EOF
        cat > OPTCELL <<EOF
100
110
000
EOF

        # 5. 定义运行函数 (超时+重试)
        run_vasp_retries() {
            local try_count=0
            local max_retries=2
            local success=0

            while [ $try_count -le $max_retries ]; do
                echo "启动 VASP (尝试 $((try_count+1)))... [超时设定: ${TASK_TIMEOUT}]"
                
                # 使用 timeout 杀死超时任务 (防止卡死)
                timeout --kill-after=5m ${TASK_TIMEOUT} mpirun -np ${VASP_PROCS} vasp_std_optcell
                local exit_code=$?

                # 超时检测
                if [ $exit_code -eq 124 ]; then
                    echo "CRITICAL: 任务超时 (${TASK_TIMEOUT})。"
                    return 124
                fi

                # Signal 9 内存溢出
                if [ $exit_code -eq 137 ] || [ $exit_code -eq 9 ]; then
                    echo "CRITICAL: 任务因内存不足 (Signal 9) 被杀。"
                    return 9
                fi

                # ZBRENT 救援
                if grep -q "ZBRENT" OUTCAR; then
                    echo "WARNING: 检测到 ZBRENT 错误。"
                    if [ $try_count -lt $max_retries ]; then
                        echo "救援: 切换到 IBRION=1 重试..."
                        cp CONTCAR POSCAR 2>/dev/null
                        sed -i 's/IBRION =.*/IBRION = 1/' INCAR
                        try_count=$((try_count+1))
                        continue
                    else
                        return 2
                    fi
                fi

                # 电子步不收敛救援
                if grep -q "self-consistency was not achieved" OUTCAR; then
                    echo "WARNING: 电子步未收敛。"
                    if [ $try_count -lt $max_retries ]; then
                        echo "救援: 增加 NELM / 减小 AMIX..."
                        cp CONTCAR POSCAR 2>/dev/null
                        sed -i 's/NELM =.*/NELM = 200/' INCAR
                        sed -i 's/AMIX =.*/AMIX = 0.05/' INCAR
                        try_count=$((try_count+1))
                        continue
                    else
                        return 3
                    fi
                fi

                # 成功检测
                if grep -q "Voluntary" OUTCAR || grep -q "General timing" OUTCAR; then
                    success=1
                    break
                else
                    echo "WARNING: VASP 异常退出。"
                    return 1
                fi
            done

            if [ $success -eq 1 ]; then return 0; else return 1; fi
        }

        # 6. 执行
        run_vasp_retries
        v_status=$?

        # 7. 记录
        if [ $v_status -eq 0 ]; then
            E_TOTAL=$(grep "energy  without" OUTCAR | tail -1 | awk '{print $4}')
            {
                echo "Calculation for ${STRUCT_LABEL} (index ${task})"
                echo "  Total Energy: ${E_TOTAL} eV"
                echo "---"
            } >> "${RESULTS_LOG}"
        elif [ $v_status -eq 124 ]; then
            {
                echo "$(date '+%Y-%m-%d %H:%M:%S') 任务超时"
                echo "  Structure: ${STRUCT_LABEL}"
                echo "---"
            } >> "${TIMEOUT_LOG}"
        else
            {
                echo "$(date '+%Y-%m-%d %H:%M:%S') 计算失败"
                echo "  Structure: ${STRUCT_LABEL}"
                echo "  Code: ${v_status}"
                echo "---"
            } >> "${FAIL_LOG}"
        fi

        cd "${ROOT_DIR}"
        return 0
    }

    # 循环
    for ((task=${BATCH_START}; task<=${BATCH_END}; task++)); do
        process_structure "${task}"
    done

    echo "批次 ${BATCH_START}-${BATCH_END} 已完成。"
    exit 0
fi

if [ $# -ne 0 ]; then
    echo "使用方法: $0            # 自动提交所有批次"
    echo "       或: $0 run <start> <end>"
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

echo "共 ${TOTAL_TASKS} 个结构，拆分 ${BATCH_COUNT} 批次。"

for ((batch=0; batch<BATCH_COUNT; batch++)); do
    BATCH_START=$(( batch * BATCH_SIZE + 1 ))
    if [ ${BATCH_START} -gt ${TOTAL_TASKS} ]; then break; fi
    BATCH_END=$(( BATCH_START + BATCH_SIZE - 1 ))
    if [ ${BATCH_END} -gt ${TOTAL_TASKS} ]; then BATCH_END=${TOTAL_TASKS}; fi

    echo "提交批次 $((batch + 1)): ${BATCH_START}-${BATCH_END}"
    sbatch "$0" run ${BATCH_START} ${BATCH_END}
    sleep 2
done

echo "全部提交完成。"
