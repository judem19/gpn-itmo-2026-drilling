#ifndef LAB4_H
#define LAB4_H

#include "variant.h"

// Признаки вина в порядке столбцов CSV:
// fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides,
// free_sulfur_dioxide, total_sulfur_dioxide, density, ph, sulphates, alcohol.
#define FEATURE_COUNT 11

struct Wine {
    double features[FEATURE_COUNT];  // пустая ячейка в CSV -> NAN
    double quality;                  // оценка качества (таргет), NAN — не указана
};

/* ============================ Чётный ISU ID ============================== */

// Vector из lab3, только элементы теперь — структуры, а не int.
struct Vector {
    struct Wine *data;  // непрерывный блок памяти под capacity вин
    int size;           // сколько вин лежит сейчас
    int capacity;       // под сколько вин выделена память
};

struct Vector *createVector(int capacity);
int appendVectorItem(struct Vector *vector, const struct Wine *wine);
void freeVector(struct Vector *vector);

/* =========================== Нечётный ISU ID ============================= */

// List из lab3, только узел теперь хранит структуру, а не int.
struct ListNode {
    struct Wine data;
    struct ListNode *next;
};

struct List {
    struct ListNode *head;  // первый узел, NULL — список пуст
    int size;               // сколько узлов в списке
};

struct List *createList(void);
int appendListItem(struct List *list, const struct Wine *wine);
void freeList(struct List *list);

/* ============================== Датасет ================================== */

// Датасет — коллекция вашего варианта. Функции ниже пишутся через Dataset.
#if ISU_ID % 2 == 0
typedef struct Vector Dataset;
#else
typedef struct List Dataset;
#endif

Dataset *readDataset(const char *path);
int writePredictions(const char *path, const double *predictions, int count);

/* ================================ kNN ==================================== */

struct KnnModel {
    Dataset *train;  // СВОЯ копия обучающей выборки
    int k;           // сколько соседей усредняем

    // Только для доп. задания (предобработка), в базовой версии не нужны:
    double fill[FEATURE_COUNT];    // чем заменять пропуски (например, медиана по train)
    double center[FEATURE_COUNT];  // нормализация признака: (x - center) / scale
    double scale[FEATURE_COUNT];
};

struct KnnModel *fitKnn(const Dataset *train, int k);
double predictKnn(const struct KnnModel *model, const struct Wine *wine);
double *predictDataset(const struct KnnModel *model, const Dataset *test);
double computeRmse(const double *predictions, const Dataset *test);
void freeKnn(struct KnnModel *model);

#endif
