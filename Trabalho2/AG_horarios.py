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
    def __init__(self, id, nome, departamento, salas_permitidas=None):
        self.id = id; self.nome = nome; self.departamento = departamento
        self.salas_permitidas = salas_permitidas or []
        
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

# --- 2. O CROMOSSOMO (VERSÃO CORRIGIDA) ---
# Esta classe foi corrigida para que o fitness "enxergue" os conflitos
class HorarioGrade:
    def __init__(self, aulas):
        self.aulas = aulas; self.fitness = 0.0; self.calcular_fitness()
        
    def calcular_fitness(self):
        penalidades_rigidas = 0
        penalidades_leves = 0 
        
        # Dicionários para contar conflitos (Lógica idêntica a analisar_conflitos)
        horarios_turmas = defaultdict(int)
        horarios_professores = defaultdict(int)
        horarios_salas = defaultdict(int)
        aulas_dia_turma_curso = defaultdict(int) 

        # IDs dos horários proibidos
        HORARIO_PROIBIDO_DEMAT_ID = 18 
        HORARIO_PROIBIDO_DECOMP_OPT_ID = 19

        for aula in self.aulas:
            # --- Checagens RÍGIDAS (Contagem de Conflitos) ---
            
            # 1. Conflitos de alocação (Turma, Prof, Sala no mesmo horário)
            horarios_turmas[(aula.turma.id, aula.horario.id)] += 1
            horarios_professores[(aula.professor.id, aula.horario.id)] += 1
            horarios_salas[(aula.sala.id, aula.horario.id)] += 1
            
            # 2. Conflitos de atributos
            if aula.sala.capacidade < aula.turma.num_alunos: 
                penalidades_rigidas += 1
            if aula.curso.id not in aula.professor.cursos_aptos: 
                penalidades_rigidas += 1
            if aula.curso.salas_permitidas and aula.sala.id not in aula.curso.salas_permitidas: 
                penalidades_rigidas += 1

            # 3. Conflitos de Regras de Negócio (Horários Proibidos)
            if aula.horario.id == HORARIO_PROIBIDO_DECOMP_OPT_ID:
                if aula.curso.departamento == 'DECOMP' or aula.curso.departamento == 'OPT':
                    penalidades_rigidas += 1 
            
            if aula.horario.id == HORARIO_PROIBIDO_DEMAT_ID:
                if aula.curso.departamento == 'DEMAT':
                    penalidades_rigidas += 1

            # --- Checagens LEVES (Preferências) ---
            aulas_dia_turma_curso[(aula.turma.id, aula.curso.id, aula.horario.dia)] += 1

        # Aplicar penalidades RÍGIDAS (com base na contagem)
        # ESTA ERA A PARTE FALTANTE NO CÓDIGO ANTERIOR
        for count in horarios_turmas.values():
            if count > 1: penalidades_rigidas += (count - 1)
        for count in horarios_professores.values():
            if count > 1: penalidades_rigidas += (count - 1)
        for count in horarios_salas.values():
            if count > 1: penalidades_rigidas += (count - 1)
            
        # Aplicar penalidades LEVES
        for count in aulas_dia_turma_curso.values():
            if count > 1: 
                penalidades_leves += (count - 1) 

        # Cálculo final do Fitness (com peso 10 para leves)
        total_penalidades = (penalidades_rigidas * 1000) + (penalidades_leves * 10)
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
                
                if curso.nome in [
                    "Extensão Universitária I", "Optativa I", 
                    "Extensão Universitária II", 
                    "Mentoria em Projetos" 
                ]:
                    num_aulas = 1
                elif curso.nome == "Algoritmos e Programação de Computadores II": 
                    num_aulas = 3
                else:
                    num_aulas = 2 
                
                prof_designado = None
                
                if curso.id == 102: 
                    prof_designado = profs_map["Prof. Daniel Rigoni"] if turma.id == 1 else profs_map["Prof. Giovane"]
                elif curso.id == 107: 
                    prof_designado = profs_map["Prof. Giovane"] if turma.id == 1 else profs_map["Prof. Cesar"]
                elif curso.id == 302: 
                    prof_designado = profs_map["Prof. Mauro"] if turma.id == 5 else profs_map["Prof. Jotair"]
                elif curso.id == 305: 
                    prof_designado = profs_map["Prof. Daniel Rigoni"] if turma.id == 5 else profs_map["Prof. Jotair"]
                elif curso.id == 401: 
                     prof_designado = profs_map["Prof. random1"] if turma.id == 7 else profs_map["Prof. random3"]
                
                if prof_designado is None:
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
        if curso.salas_permitidas: 
            return [s for s in self.dados['salas'] if s.id in curso.salas_permitidas]
        
        print(f"ALERTA: Curso {curso.nome} sem salas permitidas definidas!")
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
        HORARIO_PROIBIDO_DEMAT_ID = 18
        HORARIO_PROIBIDO_DECOMP_OPT_ID = 19
        
        if horario.id == HORARIO_PROIBIDO_DECOMP_OPT_ID and (curso.departamento == 'DECOMP' or curso.departamento == 'OPT'):
            return False 
        
        if horario.id == HORARIO_PROIBIDO_DEMAT_ID and curso.departamento == 'DEMAT':
            return False
            
        if curso.salas_permitidas and sala.id not in curso.salas_permitidas: 
            return False
            
        return (horario.id not in ocupacao['turma'][turma.id] and horario.id not in ocupacao['prof'][prof.id] and horario.id not in ocupacao['sala'][sala.id])
    
    def _ocupar_slot(self, horario, sala, turma, prof, ocupacao):
        ocupacao['turma'][turma.id].add(horario.id); ocupacao['prof'][prof.id].add(horario.id); ocupacao['sala'][sala.id].add(horario.id)
    
    def _mutacao_inteligente(self, individuo):
        if random.random() >= self.taxa_mutacao: return
        aula_para_mutar = random.choice(individuo.aulas); ocupacao = {'turma': defaultdict(set), 'prof': defaultdict(set), 'sala': defaultdict(set)}
        for a in individuo.aulas:
            if a != aula_para_mutar: self._ocupar_slot(a.horario, a.sala, a.turma, a.professor, ocupacao)
        
        salas_permitidas = self._get_salas_permitidas(aula_para_mutar.curso); 
        slots_livres = []
        
        for h in self.dados['horarios']:
            for s in salas_permitidas: 
                if self._is_slot_livre(h, s, aula_para_mutar.turma, aula_para_mutar.professor, aula_para_mutar.curso, ocupacao): 
                    slots_livres.append((h,s))
                    
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
        dias = ["SEG", "TER", "QUA", "QUI", "SEX"]; blocos = ["08:20-10:00", "10:10-11:50", "13:20-15:00", "15:10-16:50", "16:50-18:30"]
        horarios = [Horario(id=i+1, dia=dia, hora=hora) for i, (dia, hora) in enumerate(product(dias, blocos))]
        
        salas = [
            Sala(id=1, nome="Laboratório 1", capacidade=20), 
            Sala(id=2, nome="Laboratório 2", capacidade=20), 
            Sala(id=3, nome="Laboratório 3", capacidade=20), 
            Sala(id=4, nome="Laboratório 4", capacidade=20), 
            Sala(id=5, nome="Laboratório DEGEO", capacidade=20), 
            Sala(id=6, nome="Laboratório Bloco 3", capacidade=20), 
            Sala(id=7, nome="Laboratório COORTI", capacidade=20),
            Sala(id=8, nome="Bloco 1 Sala 2", capacidade=30), 
            Sala(id=9, nome="Bloco 1 Sala 3", capacidade=30)  
        ]
        lab4_id = 4
        
        lab_salas_ids = [1, 2, 3, 4, 5, 6, 7]
        demat_salas_ids = [8, 9] 
        
        
        # Tupla: (ID, DEPARTAMENTO)
        nomes_cursos_t1 = {
            "Álgebra Linear": (101, 'DEMAT'), 
            "Algoritmos e Programação de Computadores II": (102, 'DECOMP'), 
            "Arquitetura de Computadores": (103, 'DECOMP'), 
            "Cálculo I": (104, 'DEMAT'), 
            "Extensão Universitária I": (105, 'DECOMP'), 
            "Fundamentos Matemáticos para Computação": (106, 'DEMAT'), 
            "Lógica Digital": (107, 'DECOMP'), 
            "Optativa I": (108, 'OPT') 
        }
        nomes_cursos_t2 = {
            "Algoritmos e Estruturas de Dados II": (201, 'DECOMP'), 
            "Banco de Dados II": (202, 'DECOMP'), 
            "Cálculo Numérico": (203, 'DECOMP'), 
            "Engenharia de Software I": (204, 'DECOMP'), 
            "Extensão Universitária II": (205, 'DECOMP'), 
            "Interação Humano-Computador": (206, 'DECOMP'), 
            "Paradigmas em Linguagens de Programação": (207, 'DECOMP'), 
            "Programação Orientada a Objetos II": (208, 'DECOMP') 
        }
        nomes_cursos_t3 = {
            "Algoritmos em Grafos": (301, 'DECOMP'), 
            "Computação Gráfica e Processamento Digital de Imagens": (302, 'DECOMP'), 
            "Desenvolvimento para Dispositivos Móveis": (303, 'DECOMP'), 
            "Inteligência Artificial e Computacional II": (304, 'DECOMP'), 
            "Mentoria em Projetos": (305, 'DECOMP'), 
            "Projeto em Computação": (306, 'DECOMP'), 
            "Sistemas Distribuídos": (307, 'DECOMP'), 
            "Teoria da Computação": (308, 'DECOMP') 
        }
        nomes_cursos_t4 = {
            "Estágio Supervisionado": (401, 'DECOMP'), 
            "Tópicos Especiais em Ciência da Computação II": (402, 'DECOMP') 
        }

        nomes_cursos = {**nomes_cursos_t1, **nomes_cursos_t2, **nomes_cursos_t3, **nomes_cursos_t4}

        cursos = []
        for nome, (id_curso, depto) in nomes_cursos.items():
            salas_p = []
            if nome == "Desenvolvimento para Dispositivos Móveis": 
                salas_p = [lab4_id] 
            elif nome in ["Extensão Universitária I", "Extensão Universitária II"]: 
                salas_p = demat_salas_ids 
            elif depto == 'DEMAT':
                salas_p = demat_salas_ids 
            else: # DECOMP ou OPT
                salas_p = lab_salas_ids 
                
            cursos.append(Curso(id=id_curso, nome=nome, departamento=depto, salas_permitidas=salas_p))
        
        
        professores = [
            Professor(1, "Prof. Thiago Grando", [101]),        
            Professor(2, "Prof. Daniel Rigoni", [102, 305]), 
            Professor(3, "Prof. Giovane", [102, 107, 206]),      
            Professor(4, "Prof. Enrique", [103, 306]),        
            Professor(5, "Prof. Maria Regina", [104]),    
            Professor(6, "Prof. Ana Elisa", [105, 205]),    
            Professor(7, "Prof. Francine", [106]),        
            Professor(8, "Prof. Cesar", [107]),             
            Professor(9, "Prof. random1", [108, 401]),        
            Professor(10, "Prof. Luciane", [201]),          
            Professor(11, "Prof. Josiane", [202]),          
            Professor(12, "Prof. Jotair", [203, 302, 305]), 
            Professor(13, "Prof. Marcos Quinaia", [204]),   
            Professor(14, "Prof. Richard", [207]),          
            Professor(15, "Prof. Inali", [208]),            
            Professor(16, "Prof. Murilo", [301]),           
            Professor(17, "Prof. Mauro", [302]),            
            Professor(18, "Prof. Carlos", [303]),           
            Professor(19, "Prof. Angelita", [304]),         
            Professor(20, "Prof. Gisane", [307]),           
            Professor(21, "Prof. Sandra", [308]),           
            Professor(22, "Prof. random3", [401]),          
            Professor(23, "Prof. random4", [402])           
        ]
        
        ids_cursos_t1_nomes = list(nomes_cursos_t1.keys())
        ids_cursos_t2_nomes = list(nomes_cursos_t2.keys())
        ids_cursos_t3_nomes = list(nomes_cursos_t3.keys())
        ids_cursos_t4_nomes = list(nomes_cursos_t4.keys())

        ids_map = {nome: id_depto[0] for nome, id_depto in nomes_cursos.items()}
        
        turmas = [
            Turma(id=1, nome="Ciência da Computação 1° TA", cursos=[ids_map[n] for n in ids_cursos_t1_nomes], num_alunos=15), 
            Turma(id=2, nome="Ciência da Computação 1° TB", cursos=[ids_map[n] for n in ids_cursos_t1_nomes], num_alunos=15),
            Turma(id=3, nome="Ciência da Computação 2° TA", cursos=[ids_map[n] for n in ids_cursos_t2_nomes], num_alunos=15),
            Turma(id=4, nome="Ciência da Computação 2° TB", cursos=[ids_map[n] for n in ids_cursos_t2_nomes], num_alunos=15), 
            Turma(id=5, nome="Ciência da Computação 3° TA", cursos=[ids_map[n] for n in ids_cursos_t3_nomes], num_alunos=15), 
            Turma(id=6, nome="Ciência da Computação 3° TB ", cursos=[ids_map[n] for n in ids_cursos_t3_nomes], num_alunos=18),
            Turma(id=7, nome="Ciência da Computação 4° TA", cursos=[ids_map[n] for n in ids_cursos_t4_nomes], num_alunos=15),
            Turma(id=8, nome="Ciência da Computação 4° TB", cursos=[ids_map[n] for n in ids_cursos_t4_nomes], num_alunos=15) 
        ]
        
        print("Dados gerados com sucesso!"); return {'professores': professores, 'cursos': cursos, 'salas': salas, 'horarios': horarios, 'turmas': turmas}
    
    def analisar_conflitos(solucao):
        print("\n--- Análise de Conflitos da Melhor Solução ---")
        conflitos_rigidos = []
        conflitos_leves = []
        
        horarios_turmas = defaultdict(list)
        horarios_professores = defaultdict(list)
        horarios_salas = defaultdict(list)
        aulas_dia_turma_curso = defaultdict(list)
        
        HORARIO_PROIBIDO_DEMAT_ID = 18
        HORARIO_PROIBIDO_DECOMP_OPT_ID = 19
        
        for aula in solucao.aulas:
            horarios_turmas[(aula.turma.id, aula.horario.id)].append(aula)
            horarios_professores[(aula.professor.id, aula.horario.id)].append(aula)
            horarios_salas[(aula.sala.id, aula.horario.id)].append(aula)
            aulas_dia_turma_curso[(aula.turma.id, aula.curso.id, aula.horario.dia)].append(aula)
            
            if aula.sala.capacidade < aula.turma.num_alunos:
                conflitos_rigidos.append(f"[CONFLITO CAPACIDADE] Turma '{aula.turma.nome}' ({aula.turma.num_alunos}) na '{aula.sala.nome}' ({aula.sala.capacidade})")
            if aula.curso.id not in aula.professor.cursos_aptos:
                conflitos_rigidos.append(f"[CONFLITO APTIDÃO] Prof. '{aula.professor.nome}' não apto para '{aula.curso.nome}'")
            
            if aula.curso.salas_permitidas and aula.sala.id not in aula.curso.salas_permitidas:
                depto_sala = "B1" if aula.sala.id in [8,9] else "LAB"
                depto_curso = aula.curso.departamento
                conflitos_rigidos.append(f"[CONFLITO SALA RESTRITA] Curso '{aula.curso.nome}' ({depto_curso}) na sala '{aula.sala.nome}' ({depto_sala})")
            
            if aula.horario.id == HORARIO_PROIBIDO_DECOMP_OPT_ID and (aula.curso.departamento == 'DECOMP' or aula.curso.departamento == 'OPT'):
                conflitos_rigidos.append(f"[CONFLITO REGRA] Matéria '{aula.curso.nome}' ({aula.curso.departamento}) alocada no horário proibido (QUI 15:10)")
            
            if aula.horario.id == HORARIO_PROIBIDO_DEMAT_ID and aula.curso.departamento == 'DEMAT':
                 conflitos_rigidos.append(f"[CONFLITO REGRA] Matéria DEMAT '{aula.curso.nome}' alocada no horário proibido (QUI 13:20)")

        for _, aulas in horarios_turmas.items():
            if len(aulas) > 1: conflitos_rigidos.append(f"[CONFLITO TURMA] Turma '{aulas[0].turma.nome}' em conflito no horário {aulas[0].horario}")
        for _, aulas in horarios_professores.items():
            if len(aulas) > 1: conflitos_rigidos.append(f"[CONFLITO PROF] Prof. '{aulas[0].professor.nome}' em conflito no horário {aulas[0].horario}")
        for _, aulas in horarios_salas.items():
            if len(aulas) > 1: conflitos_rigidos.append(f"[CONFLITO SALA] Sala '{aulas[0].sala.nome}' em conflito no horário {aulas[0].horario}")
        
        for _, aulas in aulas_dia_turma_curso.items():
            if len(aulas) > 1: conflitos_leves.append(f"[PREFERÊNCIA] Matéria '{aulas[0].curso.nome}' (Turma: {aulas[0].turma.nome}) ocorre {len(aulas)}x na {aulas[0].horario.dia}")

        if not conflitos_rigidos: 
            print("Nenhum conflito de horário RÍGIDO encontrado. A solução é VÁLIDA!")
        else:
            print("(!) ATENÇÃO! Foram encontrados os seguintes conflitos RÍGIDOS:")
            [print(f"  - {c}") for c in conflitos_rigidos]

        if conflitos_leves:
            print("\nForam encontradas as seguintes quebras de PREFERÊNCIA (leves):")
            [print(f"  - {c}") for c in conflitos_leves]
        else:
            print("Nenhuma quebra de preferência (matéria repetida no dia) foi encontrada.")

    
    def exportar_para_matplotlib(solucao, dados):
        dias = ["SEG", "TER", "QUA", "QUI", "SEX"]
        blocos = sorted(list(set(h.hora for h in dados['horarios'])))
        
        cores_map = {
            'Álgebra Linear': '#FFADAD', 'Algoritmos e Programação de Computadores II': '#FFD6A5',
            'Arquitetura de Computadores': '#FDFFB6', 'Cálculo I': '#CAFFBF',
            'Extensão Universitária I': '#9BF6FF', 'Fundamentos Matemáticos para Computação': '#A0C4FF',
            'Lógica Digital': '#BDB2FF', 'Optativa I': '#FFC6FF',
            'Algoritmos e Estruturas de Dados II': '#FFADAD', 'Banco de Dados II': '#FFD6A5', 
            'Cálculo Numérico': '#FDFFB6', 'Engenharia de Software I': '#CAFFBF', 
            'Extensão Universitária II': '#9BF6FF', 'Interação Humano-Computador': '#A0C4FF', 
            'Paradigmas em Linguagens de Programação': '#BDB2FF', 'Programação Orientada a Objetos II': '#FFC6FF',
            'Algoritmos em Grafos': '#FFADAD', 'Computação Gráfica e Processamento Digital de Imagens': '#FFD6A5',
            'Desenvolvimento para Dispositivos Móveis': '#FDFFB6', 'Inteligência Artificial e Computacional II': '#CAFFBF',
            'Mentoria em Projetos': '#9BF6FF', 'Projeto em Computação': '#A0C4FF',
            'Sistemas Distribuídos': '#BDB2FF', 'Teoria da Computação': '#FFC6FF',
            'Estágio Supervisionado': '#FFADAD', 'Tópicos Especiais em Ciência da Computação II': '#FFD6A5'
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
                print(f"Erro ao gerar imagem '{nome_arquivo}': {e}")
            plt.close(fig)


    # --- Execução Principal (Parâmetros Corrigidos e Estáveis) ---
    DADOS_ACADEMICOS = criar_dados_com_regras_de_negocio()
    
    TAMANHO_POPULACAO = 300
    TAXA_MUTACAO = 0.15         
    TAXA_CROSSOVER = 0.85       
    NUM_ELITES = 10             
    NUM_GERACOES = 500         

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
    
    exportar_para_matplotlib(melhor_solucao, DADOS_ACADEMICOS)