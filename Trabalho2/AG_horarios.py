import random
import csv
from itertools import product
from collections import defaultdict
import matplotlib.pyplot as plt

# --- 1. CLASSES DE DADOS ---
class Professor:
    def __init__(self, id, nome, cursos_aptos):
        self.id = id; self.nome = nome; self.cursos_aptos = cursos_aptos
class Curso:
    def __init__(self, id, nome, salas_permitidas=None):
        self.id = id; self.nome = nome; self.salas_permitidas = salas_permitidas or []
class Sala:
    def __init__(self, id, nome, capacidade):
        self.id = id; self.nome = nome; self.capacidade = capacidade
class Horario:
    def __init__(self, id, dia, hora):
        self.id = id; self.dia = dia; self.hora = hora
    def __repr__(self): return f"{self.dia} {self.hora}"
class Turma:
    def __init__(self, id, nome, cursos, num_alunos):
        self.id = id; self.nome = nome; self.cursos = cursos; self.num_alunos = num_alunos
class Aula:
    def __init__(self, curso, professor, turma, sala, horario):
        self.curso = curso; self.professor = professor; self.turma = turma; self.sala = sala; self.horario = horario
    def __repr__(self): return (f"[{self.turma.nome}] {self.curso.nome} com {self.professor.nome} na {self.sala.nome} ({self.horario})")

# --- 2. O CROMOSSOMO ---
class HorarioGrade:
    def __init__(self, aulas):
        self.aulas = aulas; self.fitness = 0.0; self.calcular_fitness()
    def calcular_fitness(self):
        penalidades_rigidas = 0; penalidades_leves = 0
        conflitos_prof = defaultdict(int); conflitos_turma = defaultdict(int)
        conflitos_sala = defaultdict(int); aulas_no_mesmo_dia = defaultdict(int)
        for aula in self.aulas:
            conflitos_prof[(aula.professor.id, aula.horario.id)] += 1
            conflitos_turma[(aula.turma.id, aula.horario.id)] += 1
            conflitos_sala[(aula.sala.id, aula.horario.id)] += 1
            if aula.sala.capacidade < aula.turma.num_alunos: penalidades_rigidas += 1
            if aula.curso.id not in aula.professor.cursos_aptos: penalidades_rigidas += 1
            aulas_no_mesmo_dia[(aula.turma.id, aula.curso.id, aula.horario.dia)] += 1
            if aula.curso.salas_permitidas and aula.sala.id not in aula.curso.salas_permitidas: penalidades_rigidas += 1
        for count in conflitos_prof.values():
            if count > 1: penalidades_rigidas += (count - 1)
        for count in conflitos_turma.values():
            if count > 1: penalidades_rigidas += (count - 1)
        for count in conflitos_sala.values():
            if count > 1: penalidades_rigidas += (count - 1)
        for count in aulas_no_mesmo_dia.values():
            if count > 1: penalidades_rigidas += (count - 1)
        total_penalidades = (penalidades_rigidas * 1000) + penalidades_leves
        self.fitness = 1.0 / (1.0 + total_penalidades)

# --- 3. ALGORITMO GENÉTICO ---
class AlgoritmoGenetico:
    def __init__(self, dados, tam_populacao, taxa_mutacao, taxa_crossover, num_elites):
        self.dados = dados; self.tam_populacao = tam_populacao; self.taxa_mutacao = taxa_mutacao; self.taxa_crossover = taxa_crossover; self.num_elites = num_elites
        self.aulas_base_para_agendar = self._definir_aulas_necessarias()
        self.populacao = self._criar_populacao_inicial()
    
    def _definir_aulas_necessarias(self):
        aulas_base = []; profs_map = {p.nome: p for p in self.dados['professores']}
        for turma in self.dados['turmas']:
            for curso_id in turma.cursos:
                curso = next(c for c in self.dados['cursos'] if c.id == curso_id)
                
                if curso.nome in ["Mentoria em Projetos", "Extensão Universitária II"]:
                    num_aulas = 1
                else:
                    num_aulas = 2
                
                if curso.nome == "Computação Gráfica": 
                    prof_designado = profs_map["Prof. Mauro"] if turma.id == 1 else profs_map["Prof. Jotair"]
                elif curso.nome == "Mentoria em Projetos": 
                    prof_designado = profs_map["Prof. Daniel R."] if turma.id == 1 else profs_map["Prof. Jotair"]
                else:
                    try:
                        prof_designado = next(p for p in self.dados['professores'] if curso.id in p.cursos_aptos)
                    except StopIteration:
                        print(f"ERRO FATAL: Nenhum professor apto encontrado para o curso '{curso.nome}' (ID: {curso.id})")
                        raise Exception(f"Nenhum professor apto para {curso.nome}")
                        
                for _ in range(num_aulas): 
                    aulas_base.append({'curso': curso, 'turma': turma, 'professor': prof_designado})
        
        print(f"Total de aulas a serem agendadas: {len(aulas_base)}")
        return aulas_base
    
    def _get_salas_permitidas(self, curso):
        if curso.salas_permitidas: return [s for s in self.dados['salas'] if s.id in curso.salas_permitidas]
        return self.dados['salas']
    
    def _criar_individuo_aleatorio(self):
        aulas = []
        for aula_info in self.aulas_base_para_agendar:
            salas_permitidas = self._get_salas_permitidas(aula_info['curso'])
            aulas.append(Aula(curso=aula_info['curso'], professor=aula_info['professor'], turma=aula_info['turma'], sala=random.choice(salas_permitidas), horario=random.choice(self.dados['horarios'])))
        return HorarioGrade(aulas)
    
    def _criar_populacao_inicial(self):
        return [self._criar_individuo_aleatorio() for _ in range(self.tam_populacao)]
    
    def _selecao_torneio(self, tamanho_torneio=3):
        return max(random.sample(self.populacao, tamanho_torneio), key=lambda i: i.fitness)
    
    def _crossover_construtivo(self, pai1, pai2):
        if random.random() > self.taxa_crossover: return HorarioGrade(pai1.aulas)
        filho_aulas = []; ocupacao = {'turma': defaultdict(set), 'prof': defaultdict(set), 'sala': defaultdict(set)}; aulas_para_agendar = self.aulas_base_para_agendar[:]; random.shuffle(aulas_para_agendar)
        mapa_pai1 = {(a.turma.id, a.curso.id): [] for a in pai1.aulas}; [mapa_pai1[(a.turma.id, a.curso.id)].append(a) for a in pai1.aulas]; mapa_pai2 = {(a.turma.id, a.curso.id): [] for a in pai2.aulas}; [mapa_pai2[(a.turma.id, a.curso.id)].append(a) for a in pai2.aulas]; aulas_agendadas_por_curso = defaultdict(int)
        for aula_info in aulas_para_agendar:
            turma, curso, prof = aula_info['turma'], aula_info['curso'], aula_info['professor']; idx = aulas_agendadas_por_curso.get((turma.id, curso.id), 0)
            aulas_pai1 = mapa_pai1.get((turma.id, curso.id), [])
            if idx < len(aulas_pai1) and self._is_slot_livre(aulas_pai1[idx].horario, aulas_pai1[idx].sala, turma, prof, curso, ocupacao): aula_para_add = aulas_pai1[idx]; filho_aulas.append(aula_para_add); self._ocupar_slot(aula_para_add.horario, aula_para_add.sala, turma, prof, ocupacao)
            elif idx < len(aulas_pai2 := mapa_pai2.get((turma.id, curso.id), [])) and self._is_slot_livre(aulas_pai2[idx].horario, aulas_pai2[idx].sala, turma, prof, curso, ocupacao): aula_para_add = aulas_pai2[idx]; filho_aulas.append(aula_para_add); self._ocupar_slot(aula_para_add.horario, aula_para_add.sala, turma, prof, ocupacao)
            else:
                slot_encontrado = False; salas_permitidas = self._get_salas_permitidas(curso)
                for h_tentativa in random.sample(self.dados['horarios'], len(self.dados['horarios'])):
                    for s_tentativa in random.sample(salas_permitidas, len(salas_permitidas)):
                        if self._is_slot_livre(h_tentativa, s_tentativa, turma, prof, curso, ocupacao): nova_aula = Aula(curso, prof, turma, s_tentativa, h_tentativa); filho_aulas.append(nova_aula); self._ocupar_slot(h_tentativa, s_tentativa, turma, prof, ocupacao); slot_encontrado = True; break
                    if slot_encontrado: break
                if not slot_encontrado: aula_para_add = aulas_pai1[idx] if idx < len(aulas_pai1) else (aulas_pai2[idx] if idx < len(aulas_pai2) else Aula(curso, prof, turma, random.choice(salas_permitidas), random.choice(self.dados['horarios']))); filho_aulas.append(aula_para_add); self._ocupar_slot(aula_para_add.horario, aula_para_add.sala, turma, prof, ocupacao)
            aulas_agendadas_por_curso[(turma.id, curso.id)] = idx + 1
        return HorarioGrade(filho_aulas)
    
    def _is_slot_livre(self, horario, sala, turma, prof, curso, ocupacao):
        if curso.salas_permitidas and sala.id not in curso.salas_permitidas: return False
        return (horario.id not in ocupacao['turma'][turma.id] and horario.id not in ocupacao['prof'][prof.id] and horario.id not in ocupacao['sala'][sala.id])
    
    def _ocupar_slot(self, horario, sala, turma, prof, ocupacao):
        ocupacao['turma'][turma.id].add(horario.id); ocupacao['prof'][prof.id].add(horario.id); ocupacao['sala'][sala.id].add(horario.id)
    
    def _mutacao_inteligente(self, individuo):
        if random.random() >= self.taxa_mutacao: return
        aula_para_mutar = random.choice(individuo.aulas); ocupacao = {'turma': defaultdict(set), 'prof': defaultdict(set), 'sala': defaultdict(set)}
        for a in individuo.aulas:
            if a != aula_para_mutar: self._ocupar_slot(a.horario, a.sala, a.turma, a.professor, ocupacao)
        salas_permitidas = self._get_salas_permitidas(aula_para_mutar.curso); slots_livres = []
        for h in self.dados['horarios']:
            for s in salas_permitidas:
                if self._is_slot_livre(h, s, aula_para_mutar.turma, aula_para_mutar.professor, aula_para_mutar.curso, ocupacao): slots_livres.append((h,s))
        if slots_livres:
            novo_h, nova_s = random.choice(slots_livres); aula_para_mutar.horario = novo_h; aula_para_mutar.sala = nova_s; individuo.calcular_fitness()
    
    def executar_geracao(self):
        self.populacao.sort(key=lambda x: x.fitness, reverse=True)
        proxima_geracao = self.populacao[:self.num_elites]
        while len(proxima_geracao) < self.tam_populacao:
            pai1 = self._selecao_torneio(); pai2 = self._selecao_torneio(); filho = self._crossover_construtivo(pai1, pai2); self._mutacao_inteligente(filho); proxima_geracao.append(filho)
        self.populacao = proxima_geracao

# --- 4. EXECUÇÃO DO ALGORITMO E EXPORTAÇÃO ---
if __name__ == "__main__":
    def criar_dados_com_regras_de_negocio():
        print("Gerando conjunto de dados com regras de negócio específicas...")
        dias = ["SEG", "TER", "QUA", "QUI", "SEX"]; blocos = ["08:20-10:00", "10:10-11:50", "13:20-15:00", "15:10-16:50", "17:00-18:40"]
        horarios = [Horario(id=i+1, dia=dia, hora=hora) for i, (dia, hora) in enumerate(product(dias, blocos))]
        salas = [Sala(id=1, nome="Lab 1", capacidade=40), Sala(id=2, nome="Lab 2", capacidade=40), Sala(id=3, nome="Lab 3", capacidade=40), Sala(id=4, nome="Lab 4", capacidade=40), Sala(id=5, nome="Lab DEGEO", capacidade=40), Sala(id=6, nome="Lab Bloco 3", capacidade=40), Sala(id=7, nome="Lab COORTI", capacidade=40)]
        lab4_id = next(s.id for s in salas if s.nome == "Lab 4")
        
        nomes_cursos_t5 = {
            "IA 2": 101, "Sistemas Distribuídos": 102, "Projeto em Computação": 103, 
            "Mentoria em Projetos": 104, "Desenvolvimento Mobile": 105, "Computação Gráfica": 106, 
            "Teoria da Computação": 107, "Algoritmos em Grafos": 108
        }
        nomes_cursos_t2 = {
            "AED II": 201, "Banco de Dados II": 202, "Calculo Numérico": 203,
            "Engenharia de Software I": 204, "Extensão Universitária II": 205,
            "IHC": 206, "Paradigmas em Linguagens de Programação": 207, "POO II": 208
        }
        nomes_cursos = {**nomes_cursos_t5, **nomes_cursos_t2}

        cursos = []
        for nome, id_curso in nomes_cursos.items():
            if nome == "Desenvolvimento Mobile": cursos.append(Curso(id=id_curso, nome=nome, salas_permitidas=[lab4_id]))
            else: cursos.append(Curso(id=id_curso, nome=nome))
        
        professores = [
            Professor(1, "Prof. Carlos", [105]), 
            Professor(2, "Prof. Angelita", [101]), 
            Professor(3, "Prof. Gisane", [102]), 
            Professor(4, "Prof. Mauro", [106]), 
            Professor(5, "Prof. Jotair", [106, 104, 203]),
            Professor(6, "Prof. Daniel R.", [104]), 
            Professor(7, "Prof. Murilo", [108]), 
            Professor(8, "Prof. Sandra", [107]), 
            Professor(9, "Prof. Enrique", [103]),
            Professor(10, "Prof. Luciane", [201]),
            Professor(11, "Prof. Josiane", [202]),
            Professor(12, "Prof. Marcus Quinaia", [204]),
            Professor(13, "Prof. Ana Eliza", [205]),
            Professor(14, "Prof. Giovane", [206]),
            Professor(15, "Prof. Richard", [207]),
            Professor(16, "Prof. Inali", [208])
        ]
        
        ids_cursos_t5 = list(nomes_cursos_t5.values())
        ids_cursos_t2 = list(nomes_cursos_t2.values())
        
        turmas = [
            Turma(id=1, nome="Ciência Comp. 3° TA", cursos=ids_cursos_t5, num_alunos=17), 
            Turma(id=2, nome="Ciência Comp. 3° TB ", cursos=ids_cursos_t5, num_alunos=16),
            Turma(id=3, nome="Ciência Comp. 2° TA", cursos=ids_cursos_t2, num_alunos=15),
            Turma(id=4, nome="Eng. Software 2° TB", cursos=ids_cursos_t2, num_alunos=15)
        ]
        
        print("Dados gerados com sucesso!"); return {'professores': professores, 'cursos': cursos, 'salas': salas, 'horarios': horarios, 'turmas': turmas}
    
    def analisar_conflitos(solucao):
        print("\n--- Análise de Conflitos da Melhor Solução ---"); conflitos = []; horarios_turmas = defaultdict(list); horarios_professores = defaultdict(list); horarios_salas = defaultdict(list); aulas_dia_turma_curso = defaultdict(list)
        for aula in solucao.aulas:
            horarios_turmas[(aula.turma.id, aula.horario.id)].append(aula); horarios_professores[(aula.professor.id, aula.horario.id)].append(aula); horarios_salas[(aula.sala.id, aula.horario.id)].append(aula); aulas_dia_turma_curso[(aula.turma.id, aula.curso.id, aula.horario.dia)].append(aula)
        for _, aulas in horarios_turmas.items():
            if len(aulas) > 1: conflitos.append(f"[CONFLITO TURMA] Turma '{aulas[0].turma.nome}' em conflito no horário {aulas[0].horario}")
        for _, aulas in horarios_professores.items():
            if len(aulas) > 1: conflitos.append(f"[CONFLITO PROF] Prof. '{aulas[0].professor.nome}' em conflito no horário {aulas[0].horario}")
        for _, aulas in horarios_salas.items():
            if len(aulas) > 1: conflitos.append(f"[CONFLITO SALA] Sala '{aulas[0].sala.nome}' em conflito no horário {aulas[0].horario}")
        for _, aulas in aulas_dia_turma_curso.items():
            if len(aulas) > 1: conflitos.append(f"[CONFLITO DIA] Matéria '{aulas[0].curso.nome}' para a turma '{aulas[0].turma.nome}' ocorre {len(aulas)} vezes na {aulas[0].horario.dia}")
        if not conflitos: print("Nenhum conflito de horário rígido encontrado. A solução é VÁLIDA!")
        else:
            print("Foram encontrados os seguintes conflitos:"); [print(f"  - {c}") for c in conflitos]

    # --- FUNÇÃO REMOVIDA ---
    # def exportar_para_planilha(solucao, dados):
    #     ... (código removido) ...
    
    def exportar_para_matplotlib(solucao, dados):
        dias = ["SEG", "TER", "QUA", "QUI", "SEX"]
        blocos = sorted(list(set(h.hora for h in dados['horarios'])))
        
        cores_map = {
            # T5
            'IA 2': '#FFADAD', 'Sistemas Distribuídos': '#FFD6A5', 'Projeto em Computação': '#FDFFB6',
            'Mentoria em Projetos': '#CAFFBF', 'Desenvolvimento Mobile': '#9BF6FF', 'Computação Gráfica': '#A0C4FF',
            'Teoria da Computação': '#BDB2FF', 'Algoritmos em Grafos': '#FFC6FF',
            # T2 (Reutilizando a paleta de cores)
            'AED II': '#FFADAD', 'Banco de Dados II': '#FFD6A5', 'Calculo Numérico': '#FDFFB6',
            'Engenharia de Software I': '#CAFFBF', 'Extensão Universitária II': '#9BF6FF',
            'IHC': '#A0C4FF', 'Paradigmas em Linguagens de Programação': '#BDB2FF', 'POO II': '#FFC6FF'
        }

        for turma in dados['turmas']:
            grade = {bloco: {dia: "" for dia in dias} for bloco in blocos}
            cores_celulas = {bloco: {dia: 'w' for dia in dias} for bloco in blocos}
            
            aulas_da_turma = [a for a in solucao.aulas if a.turma.id == turma.id]
            for aula in aulas_da_turma:
                texto_celula = f"{aula.curso.nome}\n{aula.professor.nome}\n{aula.sala.nome}"
                grade[aula.horario.hora][aula.horario.dia] = texto_celula
                cores_celulas[aula.horario.hora][aula.horario.dia] = cores_map.get(aula.curso.nome, 'w')

            dados_tabela = [[grade[b][d] for d in dias] for b in blocos]
            cores_tabela = [[cores_celulas[b][d] for d in dias] for b in blocos]

            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('off')
            ax.set_title(f"Grade de Horários - {turma.nome}", fontsize=16, pad=20, weight='bold')

            tabela = ax.table(cellText=dados_tabela,
                                  rowLabels=blocos,
                                  colLabels=dias,
                                  cellColours=cores_tabela,
                                  loc='center',
                                  cellLoc='center')
            
            tabela.auto_set_font_size(False)
            tabela.set_fontsize(9)
            tabela.scale(1, 2.5)

            for (i, j), cell in tabela.get_celld().items():
                if i == 0 or j == -1:
                    cell.set_text_props(weight='bold')

            nome_arquivo = f"horario_{turma.nome.replace(' ', '_').replace('.', '')}.png"
            try:
                plt.savefig(nome_arquivo, dpi=200, bbox_inches='tight')
                print(f"Imagem '{nome_arquivo}' gerada com sucesso!")
            except Exception as e:
                print(f"Erro ao gerar a imagem '{nome_arquivo}': {e}")
            plt.close(fig)


    # --- Execução Principal ---
    DADOS_ACADEMICOS = criar_dados_com_regras_de_negocio()
    
    TAMANHO_POPULACAO = 100
    TAXA_MUTACAO = 0.15
    TAXA_CROSSOVER = 0.85
    NUM_ELITES = 10
    NUM_GERACOES = 300

    print(f"\nIniciando AG com regras de negócio específicas...")
    print(f"Parâmetros: Pop={TAMANHO_POPULACAO}, Gerações={NUM_GERACOES}, Mutação={TAXA_MUTACAO}")
    
    ag = AlgoritmoGenetico(DADOS_ACADEMICOS, TAMANHO_POPULACAO, TAXA_MUTACAO, TAXA_CROSSOVER, NUM_ELITES)
    melhor_solucao = None
    
    for geracao in range(NUM_GERACOES):
        ag.executar_geracao()
        melhor_da_geracao = ag.populacao[0]
        
        if not melhor_solucao or melhor_da_geracao.fitness > melhor_solucao.fitness:
            melhor_solucao = melhor_da_geracao
            
        if (geracao + 1) % 50 == 0:
            print(f"Geração {geracao+1:3d}: Melhor Fitness = {melhor_solucao.fitness:.5f}")
            
        if melhor_solucao.fitness == 1.0:
            print(f"\nSolução ótima encontrada na geração {geracao+1}!"); break

    print("\n" + "="*50 + "\n     MELHOR GRADE DE HORÁRIOS ENCONTRADA\n" + "="*50)
    print(f"\nFitness Final: {melhor_solucao.fitness:.5f}")
    analisar_conflitos(melhor_solucao)
    
    # Exporta apenas para a imagem
    exportar_para_matplotlib(melhor_solucao, DADOS_ACADEMICOS)