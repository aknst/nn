#ifndef NN_H
#define NN_H
#include <iostream>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <random>
#include <ctime>
#include <algorithm>

using namespace std;

random_device RD{};
mt19937 GEN{RD()};

typedef vector<double> vd;
typedef vector<vd> vvd;

vvd transpose(vvd& v) {
    if (v.size() == 0) return v;
    vvd result = vvd(
        v[0].size(),
        vd(v.size(), 0)
    );
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++)
            result[j][i] = v[i][j];
    }
    return result;
}

vvd operator +(const vvd& v1, const vvd& v2) {
    if (!(v1.size() > 0 && v2.size() > 0))
        return vvd();
    if (!(v1[0].size() == v2[0].size()))
        return vvd();
    vvd result = vvd(v1.size(), vd(v1[0].size(), 0.0));
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[i].size(); j++)
            result[i][j] = v1[i][j] + v2[i][j];
    return result;
}

vvd operator -(const vvd& v1, const vvd& v2) {
    if (!(v1.size() > 0 && v2.size() > 0))
        return vvd();
    if (!(v1[0].size() == v2[0].size()))
        return vvd();
    vvd result = vvd(v1.size(), vd(v1[0].size(), 0));
    for (int i = 0; i < result.size(); i++)
        for (int j = 0; j < result[i].size(); j++)
            result[i][j] = v1[i][j] - v2[i][j];
    return result;
}

vvd operator *(double a, const vvd v) {
    vvd result = v;
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v[i].size(); j++)
            result[i][j] *= a;
    return result;
}

vvd operator *(const vvd& v1, const vvd& v2) {
    if (v1.size() == 0) return vvd();
    if (v1[0].size() != v2.size()) return vvd();
    vvd result = vvd(v1.size(), vd(v2[0].size(), 0));
    for (int i = 0; i < v1.size(); i++)
        for (int j = 0; j < v1[i].size(); j++)
            for (int k = 0; k < v2[0].size(); k++)
                result[i][k] += v1[i][j] * v2[j][k];
    return result;
}

void fill(vvd& v, double a=-0.5, double b=0.5) {
    uniform_real_distribution<double> distr2 (a, b);
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v[i].size(); j++)
            v[i][j] = distr2(GEN);
}

int argmax(vd v) {
    int result = -1;
    double mx = -INFINITY;
    for (int i = 0; i < v.size(); i++) {
        if (v[i] > mx) {
            mx = v[i];
            result = i;
        }
    }
    return result;
}

void read_vvd(ifstream& file, vvd& v) {
    int y, x;
    file >> y >> x;
    v = vvd(y, vd(x, 0));
    for (int i = 0; i < v.size(); i++)
        for (int j = 0; j < v[i].size(); j++)
            file >> v[i][j];
}

void write_vvd(ofstream& file, vvd& v) {
    file << v.size() << ' ' << v[0].size() << endl;
    for (int i = 0; i < v.size(); i++) {
        for (int j = 0; j < v[i].size(); j++)
            file << v[i][j] << ' ';
        file << endl;
    }
}

vvd softmax(vd t) {
    vvd result;
    double m = -INFINITY;
    for (int i = 0; i < t.size(); i++)
        if (m < t[i]) m = t[i];
    double sum = 0.0;
    for (int i = 0; i < t.size(); i++)
        sum += exp(t[i] - m);
    double constant = m + log(sum);
    for (int i = 0; i < t.size(); i++)
        t[i] = exp(t[i] - constant);
    result.push_back(t);
    return result;
}

struct neural_net {
    /* Поля */
    vvd current_input;
    vvd weights, bias;
    vvd delta_weights, delta_bias;
    double learning_rate;
    bool ready;

    /* Конструкторы */
    neural_net() {};
    neural_net(int N, int M, double a);
    neural_net(string filepath, double a);

    /* Методы */
    vvd predict(vvd input);
    void calc_grad(vvd z, vvd ans);
    void update_model(vector<vvd>& inputs, vector<vvd>& answers);
    double accuracy(vector<vvd>& inputs, vector<vvd>& answers);
    void save_model(string filepath);
};

neural_net::neural_net(int N, int M, double a=0.01) {
    weights = vvd(N, vd(M, 0.0));
    delta_weights = weights;
    fill(weights);
    bias  = vvd(1, vd(M, 0.0));
    delta_bias = bias;
    fill(bias);
    learning_rate = a;
    ready = true;
}

neural_net::neural_net(string filepath, double a=0.01) {
    ifstream file(filepath);
    learning_rate = a;
    read_vvd(file, weights);
    read_vvd(file, bias);
    delta_weights = vvd(weights.size(), vd(weights[0].size(), 0));
    delta_bias = vvd(bias.size(), vd(bias[0].size(), 0));
    file.close();
    ready = true;
}

vvd neural_net::predict(vvd input) {
    current_input = input;
    vvd t = input * weights + bias;
    vvd z = softmax(t[0]);
    return z;
}

void neural_net::calc_grad(vvd z, vvd ans) {
    vvd dL_db = z - ans;
    vvd dL_dW = transpose(current_input)*dL_db;
    delta_bias = delta_bias + dL_db;
    delta_weights = delta_weights + dL_dW;
}

void neural_net::update_model(vector<vvd>& inputs, vector<vvd>& answers) {
    int batch_size = inputs.size();
    fill(delta_weights, 0, 0);
    fill(delta_bias, 0, 0);
    for (int i = 0; i < batch_size; i++) {
        vvd z = predict(inputs[i]);
        calc_grad(z, answers[i]);
    }
    double k = 1.0/batch_size;
    delta_weights = k*delta_weights;
    weights = weights - learning_rate*delta_weights;
    delta_bias = k*delta_bias;
    bias = bias - learning_rate*delta_bias;
}

double neural_net::accuracy(vector<vvd>& inputs, vector<vvd>& answers) {
    double n = 0;
    for (int i = 0; i < inputs.size(); i++) {
        vvd z = predict(inputs[i]);
        if (argmax(z[0]) == argmax(answers[i][0]))
            n += 1.0;
    }
    return n / inputs.size();
}

void neural_net::save_model(string filepath="model.txt") {
    ofstream file(filepath);
    write_vvd(file, weights);
    write_vvd(file, bias);
    file.close();
}
#endif // NN_H
