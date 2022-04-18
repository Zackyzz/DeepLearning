#lang racket/gui
(require plot)

(define (file-lines->list path)
  (call-with-input-file path
    (λ (file)
      (for/list ([line (in-lines file)])
        (map string->number (string-split line))))))

(define dataset (take (file-lines->list "circled.txt") 5000))

;-------------------------PREPROCESS------------------------------------

(define (change-class dataset prev after)
  (map (λ(x) (if (= prev (car x)) (cons after (cdr x)) x)) dataset))

(set! dataset (change-class (change-class dataset 3 1) 4 2))

(define (get-max lst)
  (apply max (map first lst)))

(define nr-zones (+ 1 (get-max dataset)))
(define dimension 2)

(define (codify target)
  (for/list ([i (in-range nr-zones)])
    (if (= i target) 1 0)))

(define (build lst)
  (for/list ([i (in-list lst)])
    (cons (codify (first i)) (map (λ(x) (exact->inexact (/ x 300))) (rest i)))))

(define items (build dataset))

;-------------------------BUILD NETWORK--------------------------------

(define (none x) x)
(define (sigmoid x) (/ 1 (+ 1 (exp (- x)))))
(define (d->sigmoid x) (* (sigmoid x) (- 1 (sigmoid x))))
(define SIGMOID (list sigmoid d->sigmoid))

(define (relu x) (if (< x 0) (* 0.01 x) x))
(define (d->relu x) (if (< x 0) 0.01 1))
(define RELU (list relu d->relu))

(define (selu x) (if (< x 0) (* 1.758094282 (sub1 (exp x))) (* 1.0507 x)))
(define (d->selu x) (if (< x 0) (* 1.758094282 (exp x)) 1.0507))
(define SELU (list selu d->selu))

(define (dropr x) (drop-right x 1))
(define (transpose matrix) (apply map list matrix))
(define (dot x y) (apply + (map * x y)))

(define (build-network . structure)
  (let ([shape (map first structure)] [activation-functions (map second structure)])
    (for/list ([i (dropr shape)] [j (rest shape)] [f (rest activation-functions)])
      (list (build-list j (λ _ (build-list i (λ _ (/ (- (* 2 (random)) 1) (sqrt i))))))
            (build-list j (λ _ (/ (- (* 2 (random)) 1) (sqrt i))))
            f))))

(define (forward input layer)
  (let ([output (map + (map (λ(x) (apply + (map * x input))) (first layer)) (second layer))])
    (list (map (first (third layer)) output) (map (second (third layer)) output))))

(define (feed-forward input network [outputs empty])
  (cond
    [(empty? network) outputs]
    [else
     (define result (forward input (first network)))
     (feed-forward (first result) (rest network) (cons result outputs))]))

(define (MSE output target)
  (* 0.5 (apply + (map (λ(x y) (sqr (- x y))) (first output) target))))

(define (d/MSE output target)
  (map * (map - (first output) target) (second output)))

(define (d/cross-entropy output target)
  (map - (first output) target))

(define (get-deltas weights outputs delta)
  (if (empty? weights)
      (list delta)
      (cons delta
            (get-deltas (rest weights) (rest outputs)
                        (map * (first outputs)
                             (map (λ(x) (dot x delta)) (transpose (first weights))))))))

(define (get-derivatives deltas outputs)
  (for/list ([i deltas] [j outputs])
    (list (map (λ(y) (map (λ(x) (* x y)) j)) i) i)))

(define (update x y [step a])
  (map (λ(x y) (- x (* step y))) x y))

(define (limited-update x y [step a])
  (map (λ(x y) (- (* 0.999999 x) (* step y))) x y))

(define (update-layer layer layer-derivatives)
  (list (map update (first layer) (first layer-derivatives))
        (update (second layer) (second layer-derivatives))
        (third layer)))

(define (update-network network derivatives)
  (map update-layer network derivatives))

(define (go-epoch inputs network [Error 0])
  (cond
    [(empty? inputs) (list Error network)]
    [else
     (define input (first inputs))
     (define results (feed-forward (rest input) network))
     (define output (first results))
     (define deltas (get-deltas 
                     (dropr (reverse (map first network))) (rest (map second results))
                     (d/cross-entropy output (first input))))
     (define dxs (get-derivatives deltas (append (rest (map first results)) (list (rest input)))))
     (go-epoch (rest inputs) (update-network network (reverse dxs))
               (+ Error (MSE output (first input))))]))

(define a 0.01)
(define network (build-network (list dimension none)
                               (list 6 SELU) (list 5 SELU)
                               (list 6 SELU) (list nr-zones SIGMOID)))


;;---------------------------TEST NETWORK-------------------------------------

(define all-points
  (apply append
         (for/list ([i (in-range -300 300 5)])
           (for/list ([j (in-range -300 300 5)])
             (map (λ(x) (/ x 300.0)) (list i j))))))

(define (test input network)
  (caar (feed-forward input network)))

(define (decodify lst to-find)
  (for/or ([element lst] [i (in-naturals)] #:when (equal? to-find element)) i))

(define (test-set points network)
  (for/list ([i (in-list points)])
    (define result (test i network))
    (cons (decodify result (apply max result)) (map (λ(x) (* x 300)) i))))

(define (retrieve-points lst zone)
  (if (empty? lst)
      empty
      (if (= (caar lst) zone)
          (cons (cdar lst) (retrieve-points (cdr lst) zone))
          (retrieve-points (cdr lst) zone))))

;;---------------------------PLOT-------------------------------------

(define colors (list "DarkGreen" "Navy" "Magenta" "Red" "SlateGray"
                     "DeepPink" "RoyalBlue" "Magenta" "Coral"))

(define main-window (new frame% [label "DL"] [width 700] [height 700] [x 0] [y 0]))
(define canvas-panel (new panel% [parent main-window]))
(define function-canvas (new canvas% [parent canvas-panel]))

(send main-window show #t)
(plot-background-alpha 0)
(sleep/yield 0)

(define background
  (plot-bitmap
   (for/list ([i (in-range nr-zones)])
     (points (retrieve-points dataset i)
             #:x-min -300 #:x-max 300
             #:y-min -300 #:y-max 300
             #:sym 'fullcircle #:color (list-ref colors i)
             #:alpha 1))
   #:width (- (send canvas-panel get-width) 0)
   #:height (- (send canvas-panel get-height) 0)))

(define (randomize network)
  (map (λ(x) (list (map shuffle (first x)) (second x))) network))

(define (plot-epochs inputs network [n 1] [Error 0])
  (cond
    [(= n 100) (list Error network)]
    [else
     (define epoch-results (go-epoch inputs network))
     (define tested (test-set all-points (second epoch-results)))
     (define dc (send function-canvas get-dc))
     (send dc clear)
     (send dc draw-bitmap background 0 0)
     (plot/dc (list
               (for/list ([i (in-range nr-zones)])
                 (points (retrieve-points tested i)
                         #:x-min -300 #:x-max 300
                         #:y-min -300 #:y-max 300
                         #:sym 'fullcircle #:color (list-ref colors i)
                         #:alpha 0.3)))
              dc
              0 0
              (- (send canvas-panel get-width) 0)
              (- (send canvas-panel get-height) 0))
     (when (> n 0) (printf "Epoch: ~a -> Error = ~a\n" n Error))
     (plot-epochs inputs (second epoch-results) (+ n 1) (first epoch-results))]))

(plot-epochs items network)